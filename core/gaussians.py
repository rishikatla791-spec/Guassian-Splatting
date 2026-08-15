"""
GaussianModel: Core learnable 3D Gaussian scene representation.

Each Gaussian is parametrized by:
  μ  ∈ ℝ³        position (xyz)
  q  ∈ ℝ⁴        unit quaternion [w,x,y,z], stored raw → normalized on use
  s  ∈ ℝ³        log-scale → exp(s) gives actual axis half-lengths
  α  ∈ ℝ         logit-opacity → sigmoid(α) ∈ (0,1)
  f  ∈ ℝ^(K×3)   SH coefficients, K = (sh_degree+1)²

Activation functions:
  scale:   exp(s)        ∈ (0, ∞)
  opacity: sigmoid(α)    ∈ (0, 1)
  color:   SH eval + 0.5 → clamped to [0, 1]

Densification (Kerbl et al. §4.3):
  - Under-reconstruction: clone (small) or split (large) Gaussians
    with high 2D positional gradient magnitude
  - Over-reconstruction: prune transparent, too-large Gaussians

Mathematical correctness:
  - Scale init: log(mean_knn_dist) for each point (not arbitrary)
  - Opacity init: inverse_sigmoid(0.1) ≈ -2.197
  - Split scale: new_scale = log(exp(old_scale) / (φ·n_splits))
    where φ=0.8 is the splitting factor from the paper
  - Cloning: exact duplicate at same position (optimizer state zeroed)
  - Pruning: boolean mask applied consistently to all parameter tensors
    and their optimizer state
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sh import RGB2SH, num_sh_coefficients
from .math_utils import inverse_sigmoid, safe_log


# ══════════════════════════════════════════════════════════════════
# Exponential LR Schedule
# ══════════════════════════════════════════════════════════════════

def get_expon_lr_func(
    lr_init: float,
    lr_final: float,
    lr_delay_steps: int = 0,
    lr_delay_mult: float = 1.0,
    max_steps: int = 1_000_000,
):
    """
    Exponential learning rate decay schedule with optional warm-up.

    lr(t) = lr_init · (lr_final / lr_init)^(t/T)
           = lr_init · exp(t/T · log(lr_final/lr_init))

    With warmup (lr_delay_steps > 0):
        delay_rate = lr_delay_mult + (1 - lr_delay_mult) · sin(π·min(t, D)/(2D))
        lr(t)      = delay_rate · lr_base(t)

    The sine warmup provides smooth, continuous LR increase from
    lr_init * lr_delay_mult to lr_init over lr_delay_steps steps.

    Args:
        lr_init:         initial learning rate
        lr_final:        final learning rate
        lr_delay_steps:  number of warmup steps (0 = no warmup)
        lr_delay_mult:   LR multiplier at step 0 during warmup
        max_steps:       total steps over which to decay

    Returns:
        callable step → float
    """
    if lr_init == 0.0 and lr_final == 0.0:
        return lambda step: 0.0

    def func(step: int) -> float:
        if step < 0:
            return 0.0

        # Warmup: sine ramp from lr_delay_mult to 1.0
        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1.0 - lr_delay_mult) * math.sin(
                0.5 * math.pi * min(step, lr_delay_steps) / lr_delay_steps
            )
        else:
            delay_rate = 1.0

        # Exponential decay
        t = min(step / max(max_steps, 1), 1.0)
        log_lerp = lr_init * (lr_final / lr_init) ** t
        return delay_rate * log_lerp

    return func


# ══════════════════════════════════════════════════════════════════
# GaussianModel
# ══════════════════════════════════════════════════════════════════

class GaussianModel(nn.Module):
    """
    Learnable 3D Gaussian scene representation.

    All parameters are stored as nn.Parameter and directly optimized.
    The model supports progressive SH degree increase and adaptive
    densification/pruning.
    """

    def __init__(self, sh_degree: int = 3):
        """
        Args:
            sh_degree: maximum SH degree for view-dependent color (0..3)
        """
        super().__init__()
        assert 0 <= sh_degree <= 3, f"sh_degree must be in [0,3], got {sh_degree}"

        self.sh_degree = sh_degree
        self.active_sh_degree = 0    # progressively increased during training
        self.max_sh_degree = sh_degree

        # Parameters (initialized by init_from_pointcloud or load_ply)
        self._xyz: nn.Parameter            # (N, 3)
        self._features_dc: nn.Parameter   # (N, 1, 3)
        self._features_rest: nn.Parameter  # (N, K-1, 3)
        self._scaling: nn.Parameter        # (N, 3) log-scale
        self._rotation: nn.Parameter       # (N, 4) raw quaternion
        self._opacity: nn.Parameter        # (N, 1) logit-opacity

        # Densification bookkeeping (not learnable)
        self.xyz_gradient_accum: torch.Tensor  # (N, 1)
        self.denom: torch.Tensor               # (N, 1) int32
        self.max_radii2D: torch.Tensor         # (N,)

    # ──────────────────────────────────────────────────────────────
    # Initialization
    # ──────────────────────────────────────────────────────────────

    def init_from_pointcloud(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        spatial_lr_scale: float = 1.0,
        knn_k: int = 3,
    ) -> None:
        """
        Initialize Gaussians from a sparse point cloud.

        Scale initialization:
            For each point p, compute the mean distance to its k nearest
            neighbors. This gives an isotropic scale estimate that adapts
            to local point density:
                s = log(mean_knn_dist)  (so exp(s) = mean_knn_dist)

            This is mathematically sound: a Gaussian placed at p with
            σ = mean_knn_dist approximately covers the Voronoi cell of p.

        Opacity initialization:
            α = inverse_sigmoid(0.1) ≈ -2.197
            → starts slightly transparent to encourage early densification

        Color initialization:
            SH DC = RGB2SH(color) = (color - 0.5) / C₀
            SH rest = 0  (starts view-independent)

        Rotation initialization:
            Identity quaternion [1, 0, 0, 0]

        Args:
            points:           (N, 3) float32/64 point positions
            colors:           (N, 3) float32 RGB colors in [0,1], or None
            spatial_lr_scale: multiplier for position LR (set to scene extent)
            knn_k:            k nearest neighbors for scale initialization
        """
        N = points.shape[0]
        print(f"[GaussianModel] Initializing {N:,} Gaussians from point cloud...")

        pts = torch.tensor(points, dtype=torch.float32)

        # ── Scale: k-NN mean distance ──────────────────────────────
        # Chunked to avoid O(N²) memory for large point clouds
        chunk = min(N, 8192)
        log_scale_list = []
        for i in range(0, N, chunk):
            batch = pts[i:i+chunk]
            dists = torch.cdist(batch, pts)       # (chunk, N)
            dists[:, i:i+chunk].fill_diagonal_(1e9)  # exclude self
            knn_dists, _ = dists.topk(knn_k, largest=False, dim=-1)  # (chunk, k)
            mean_d = knn_dists.mean(dim=-1, keepdim=True).clamp(min=1e-7)  # (chunk, 1)
            log_scale_list.append(torch.log(mean_d).expand(-1, 3))  # (chunk, 3)
        log_scale = torch.cat(log_scale_list, dim=0)  # (N, 3)

        # ── Rotation: identity quaternion [w=1, x=y=z=0] ──────────
        rots = torch.zeros(N, 4)
        rots[:, 0] = 1.0

        # ── Opacity: inverse_sigmoid(0.1) ─────────────────────────
        init_opacity = 0.1
        opacities = torch.full(
            (N, 1),
            math.log(init_opacity / (1.0 - init_opacity)),
            dtype=torch.float32
        )

        # ── Color: SH coefficients ─────────────────────────────────
        K = num_sh_coefficients(self.max_sh_degree)
        if colors is None:
            colors = np.full((N, 3), 0.5, dtype=np.float32)

        colors_t = torch.tensor(colors, dtype=torch.float32).clamp(0.0, 1.0)
        fdc   = RGB2SH(colors_t).unsqueeze(1)    # (N, 1, 3)
        frest = torch.zeros(N, K - 1, 3)         # (N, K-1, 3) — zero-initialized

        # ── Register parameters ────────────────────────────────────
        self._xyz           = nn.Parameter(pts)
        self._features_dc   = nn.Parameter(fdc)
        self._features_rest = nn.Parameter(frest)
        self._scaling       = nn.Parameter(log_scale)
        self._rotation      = nn.Parameter(rots)
        self._opacity       = nn.Parameter(opacities)

        self._init_densification_stats()
        print(f"[GaussianModel] Done. N={N:,}")

    def _init_densification_stats(self) -> None:
        """Reset densification bookkeeping tensors to zero."""
        N = self.num_gaussians
        device = self._xyz.device
        self.xyz_gradient_accum = torch.zeros(N, 1, device=device, dtype=torch.float32)
        self.denom              = torch.zeros(N, 1, device=device, dtype=torch.int32)
        self.max_radii2D        = torch.zeros(N,    device=device, dtype=torch.float32)

    def to(self, device: torch.device | str) -> GaussianModel:
        super().to(device)
        if hasattr(self, "xyz_gradient_accum") and self.xyz_gradient_accum is not None:
            self.xyz_gradient_accum = self.xyz_gradient_accum.to(device)
            self.denom              = self.denom.to(device)
            self.max_radii2D        = self.max_radii2D.to(device)
        return self

    # ──────────────────────────────────────────────────────────────
    # Activated properties
    # ──────────────────────────────────────────────────────────────

    @property
    def get_xyz(self) -> torch.Tensor:
        """(N, 3) Gaussian centers in world space (no activation — position is raw)."""
        return self._xyz

    @property
    def get_features(self) -> torch.Tensor:
        """(N, K, 3) full SH feature tensor (DC + rest bands)."""
        return torch.cat([self._features_dc, self._features_rest], dim=1)

    @property
    def get_opacity(self) -> torch.Tensor:
        """(N, 1) opacity in (0, 1) via sigmoid activation."""
        return torch.sigmoid(self._opacity)

    @property
    def get_scaling(self) -> torch.Tensor:
        """(N, 3) scale values in (0, ∞) via exp activation."""
        return torch.exp(self._scaling)

    @property
    def get_rotation(self) -> torch.Tensor:
        """(N, 4) unit quaternions [w, x, y, z] via L2 normalization."""
        return F.normalize(self._rotation, p=2, dim=-1)

    @property
    def num_gaussians(self) -> int:
        """Current number of Gaussians."""
        if hasattr(self, '_xyz'):
            return self._xyz.shape[0]
        return 0

    def get_covariance(self, scale_modifier: float = 1.0) -> torch.Tensor:
        """
        Compute 3D covariance upper-triangular (N, 6).
        Σ = R·diag(s)·diag(s)·Rᵀ = R·S²·Rᵀ
        """
        from .math_utils import build_covariance_3d
        return build_covariance_3d(self._scaling, scale_modifier, self.get_rotation)

    # ──────────────────────────────────────────────────────────────
    # SH degree management
    # ──────────────────────────────────────────────────────────────

    def oneupSHdegree(self) -> None:
        """Increment active SH degree by 1 (up to max_sh_degree)."""
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            print(f"[GaussianModel] SH degree increased to {self.active_sh_degree}")

    # ──────────────────────────────────────────────────────────────
    # Densification
    # ──────────────────────────────────────────────────────────────

    def add_densification_stats(
        self,
        viewspace_point_tensor: torch.Tensor,
        update_filter: torch.Tensor,
    ) -> None:
        """
        Accumulate 2D gradient norms for densification criterion.

        The densification signal is |∇_xy L| — the magnitude of the
        loss gradient w.r.t. the 2D projected position of each Gaussian.
        Large gradients indicate Gaussians that need to cover more area.

        Accumulated separately for each Gaussian and divided by view count
        in densify_and_prune to get the per-Gaussian average.

        Args:
            viewspace_point_tensor: (N, 3) screen-space proxy (requires_grad=True)
            update_filter:          (N,) bool mask of visible Gaussians
        """
        if viewspace_point_tensor.grad is None:
            return
        grad_xy = viewspace_point_tensor.grad[update_filter, :2]  # (M, 2)
        self.xyz_gradient_accum[update_filter] += grad_xy.norm(dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def densify_and_prune(
        self,
        max_grad: float,
        min_opacity: float,
        extent: float,
        max_screen_size: float,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, int]:
        """
        Adaptive Gaussian densification and pruning (Kerbl et al. §4.3).

        Algorithm:
        ──────────
        1. Compute average 2D gradient norm per Gaussian:
               avg_grad = xyz_gradient_accum / max(denom, 1)

        2. Densification criterion: avg_grad ≥ max_grad (τ_pos in paper)
           Under-reconstructed Gaussians need more resolution.

        3. Clone vs. Split:
           - Clone: max(s) ≤ ε_clone = 0.01 * extent
             Small Gaussians → duplicate at same position (cover small features)
           - Split: max(s) > ε_split = 0.01 * extent
             Large Gaussians → split into 2, sampled from distribution

        4. Pruning:
           - Low opacity: σ(α) < min_opacity (nearly invisible)
           - Large 2D radius: max_radii2D > max_screen_size (covering too much screen)
           - Large 3D scale: max(s) > 0.1 * extent (too large globally)

        Args:
            max_grad:        2D gradient threshold (τ_pos)
            min_opacity:     opacity pruning threshold
            extent:          scene extent (used for scale thresholds)
            max_screen_size: maximum allowed 2D bounding radius (pixels)
            optimizer:       Adam optimizer

        Returns:
            dict with counts: n_cloned, n_split, n_pruned
        """
        # Compute average gradient per Gaussian
        denom_f = self.denom.float().clamp(min=1)
        avg_grad = self.xyz_gradient_accum / denom_f  # (N, 1)
        avg_grad[avg_grad.isnan()] = 0.0

        selected_mask = avg_grad.squeeze() >= max_grad  # (N,) bool

        scale_threshold = 0.01 * extent

        # Clone: selected AND small scale
        clone_mask = selected_mask & (
            self.get_scaling.max(dim=1).values <= scale_threshold
        )
        n_cloned = int(clone_mask.sum().item())
        self._clone_gaussians(clone_mask, optimizer)

        # After cloning: need to extend mask for new Gaussians
        # Cloned Gaussians start with zero gradient → will not be split
        selected_extended = torch.cat([
            selected_mask,
            torch.zeros(n_cloned, dtype=torch.bool, device=self._xyz.device)
        ])

        # Split: selected AND large scale
        split_mask = selected_extended & (
            self.get_scaling.max(dim=1).values > scale_threshold
        )
        n_split = int(split_mask.sum().item())
        self._split_gaussians(split_mask, optimizer)

        # Prune: transparent, oversized
        prune_mask = self.get_opacity.squeeze() < min_opacity
        if max_screen_size > 0:
            big_screen = self.max_radii2D > max_screen_size
            big_world  = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = prune_mask | big_screen | big_world
        n_pruned = int(prune_mask.sum().item())
        self._prune_gaussians(prune_mask, optimizer)

        # Reset stats after densification
        self._init_densification_stats()
        torch.cuda.empty_cache()

        print(f"[Densify] Cloned={n_cloned}, Split={n_split}, Pruned={n_pruned} -> N={self.num_gaussians:,}")
        return {"n_cloned": n_cloned, "n_split": n_split, "n_pruned": n_pruned}

    def _clone_gaussians(
        self,
        mask: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """
        Clone Gaussians selected by mask.

        Creates exact duplicates at the same position. The optimizer state
        for new Gaussians is initialized to zero (fresh Adam moments).
        This allows the clones to quickly diverge from their source.
        """
        if mask.sum() == 0:
            return
        new_tensors = {
            "_xyz":           self._xyz[mask].detach(),
            "_features_dc":   self._features_dc[mask].detach(),
            "_features_rest": self._features_rest[mask].detach(),
            "_opacity":       self._opacity[mask].detach(),
            "_scaling":       self._scaling[mask].detach(),
            "_rotation":      self._rotation[mask].detach(),
        }
        self._cat_tensors_to_optimizer(optimizer, new_tensors)

    def _split_gaussians(
        self,
        mask: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        n_splits: int = 2,
    ) -> None:
        """
        Split large Gaussians into n_splits smaller ones.

        New position: sampled from original Gaussian distribution
            δ ~ N(0, S²)  in local frame
            new_μ = R δ + μ  (rotate offset to world space)

        New scale (log-space):
            new_log_s = log(exp(log_s) / (φ · n_splits))
                      = log_s - log(φ · n_splits)
            where φ = 0.8 (splitting factor from Kerbl et al.)

        The factor 0.8 < 1 ensures the two new Gaussians are slightly
        smaller than the original, preventing infinite growth cycles.

        New rotation, color, opacity: inherited from parent.
        """
        N_split = int(mask.sum().item())
        if N_split == 0:
            return

        from .math_utils import quaternion_to_rotation_matrix

        scales   = self.get_scaling[mask]    # (N_split, 3)
        rots_q   = self.get_rotation[mask]   # (N_split, 4)
        centers  = self._xyz[mask]           # (N_split, 3)

        # Sample Gaussian offsets for all splits
        stds = scales.repeat(n_splits, 1)   # (N_split*n_splits, 3)
        samples = torch.normal(mean=torch.zeros_like(stds), std=stds)  # (N_split*n_splits, 3)

        # Rotate offsets to world space
        R = quaternion_to_rotation_matrix(rots_q).repeat(n_splits, 1, 1)  # (N_split*n, 3, 3)
        offsets = (R @ samples.unsqueeze(-1)).squeeze(-1)  # (N_split*n, 3)
        new_xyz = centers.repeat(n_splits, 1) + offsets    # (N_split*n, 3)

        # New scale: divide by (0.8 * n_splits), in log-space = subtract
        log_scale_reduction = math.log(0.8 * n_splits)
        new_scaling = self._scaling[mask].repeat(n_splits, 1) - log_scale_reduction

        new_rotation  = self._rotation[mask].repeat(n_splits, 1)
        new_fdc       = self._features_dc[mask].repeat(n_splits, 1, 1)
        new_frest     = self._features_rest[mask].repeat(n_splits, 1, 1)
        new_opacity   = self._opacity[mask].repeat(n_splits, 1)

        self._cat_tensors_to_optimizer(optimizer, {
            "_xyz":           new_xyz,
            "_features_dc":   new_fdc,
            "_features_rest": new_frest,
            "_opacity":       new_opacity,
            "_scaling":       new_scaling,
            "_rotation":      new_rotation,
        })
        # Remove original (now replaced by splits)
        self._prune_gaussians(mask, optimizer)

    def _prune_gaussians(
        self,
        mask: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """
        Remove Gaussians selected by boolean mask (True = remove).
        Updates optimizer state tensors to match new parameter sizes.
        """
        keep = ~mask
        # Save densification stats for kept Gaussians
        grad_acc_new = self.xyz_gradient_accum[keep]
        denom_new    = self.denom[keep]
        radii_new    = self.max_radii2D[keep]

        self._prune_optimizer(optimizer, keep)

        self.xyz_gradient_accum = grad_acc_new
        self.denom              = denom_new
        self.max_radii2D        = radii_new

    # ──────────────────────────────────────────────────────────────
    # Optimizer tensor management
    # ──────────────────────────────────────────────────────────────

    def _cat_tensors_to_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
        tensors_dict: Dict[str, torch.Tensor],
    ) -> None:
        """
        Concatenate new tensors to existing parameters and extend Adam states.

        Adam states (exp_avg, exp_avg_sq) for new Gaussians are initialized
        to zero, which is equivalent to starting fresh Adam optimization
        for those parameters.

        This is the only correct way to add parameters mid-training without
        corrupting the optimizer's running statistics.
        """
        for group in optimizer.param_groups:
            name = group.get("name")
            if name not in tensors_dict:
                continue

            param = group["params"][0]
            ext   = tensors_dict[name]

            # Concatenate along first dimension (N dimension)
            new_data  = torch.cat([param.data.detach(), ext], dim=0)
            new_param = nn.Parameter(new_data, requires_grad=True)

            # Extend Adam state with zeros for new elements
            stored_state = optimizer.state.get(param)
            if stored_state is not None:
                zeros = torch.zeros(ext.shape, device=ext.device, dtype=ext.dtype)
                stored_state["exp_avg"]    = torch.cat([stored_state["exp_avg"],    zeros], dim=0)
                stored_state["exp_avg_sq"] = torch.cat([stored_state["exp_avg_sq"], zeros], dim=0)
                if "max_exp_avg_sq" in stored_state:  # AMSGrad
                    stored_state["max_exp_avg_sq"] = torch.cat(
                        [stored_state["max_exp_avg_sq"], zeros], dim=0
                    )
                optimizer.state.pop(param)
                optimizer.state[new_param] = stored_state

            group["params"][0] = new_param
            setattr(self, name, new_param)

    def _prune_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
        keep: torch.Tensor,
    ) -> None:
        """
        Slice optimizer parameter tensors and Adam states to match kept Gaussians.
        """
        param_map = {
            id(self._xyz): "_xyz",
            id(self._features_dc): "_features_dc",
            id(self._features_rest): "_features_rest",
            id(self._opacity): "_opacity",
            id(self._scaling): "_scaling",
            id(self._rotation): "_rotation",
        }

        for group in optimizer.param_groups:
            param = group["params"][0]
            name = group.get("name") or param_map.get(id(param))
            new_param = nn.Parameter(param.data.detach()[keep], requires_grad=True)

            stored_state = optimizer.state.get(param)
            if stored_state is not None:
                stored_state["exp_avg"]    = stored_state["exp_avg"][keep]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][keep]
                if "max_exp_avg_sq" in stored_state:
                    stored_state["max_exp_avg_sq"] = stored_state["max_exp_avg_sq"][keep]
                optimizer.state.pop(param)
                optimizer.state[new_param] = stored_state

            group["params"][0] = new_param
            if name:
                setattr(self, name, new_param)

    # ──────────────────────────────────────────────────────────────
    # Opacity reset
    # ──────────────────────────────────────────────────────────────

    def reset_opacity(
        self,
        optimizer: torch.optim.Optimizer,
        reset_value: float = 0.01,
    ) -> None:
        """
        Reset all opacities to min(current_opacity, reset_value).

        This is a key regularization in 3DGS training: by periodically
        resetting opacities, we force Gaussians to re-compete for coverage.
        Floaters (spurious Gaussians) typically fail to recover and get
        pruned in the next densification step.

        Mathematical operation:
            new_α_raw = logit(min(sigmoid(α_raw), reset_value))

        Args:
            reset_value: target maximum opacity after reset (default 0.01)
        """
        current_opacity = self.get_opacity                              # (N, 1) sigmoid
        clipped_opacity = torch.min(
            current_opacity,
            torch.full_like(current_opacity, reset_value)
        )  # (N, 1)
        # Convert back to logit space (numerically stable)
        new_logit = torch.log(clipped_opacity / (1.0 - clipped_opacity + 1e-10))

        for group in optimizer.param_groups:
            if group.get("name") == "_opacity":
                param = group["params"][0]
                new_param = nn.Parameter(new_logit.detach())
                # Reset Adam state for opacity (since values changed dramatically)
                stored_state = optimizer.state.get(param)
                if stored_state is not None:
                    optimizer.state.pop(param)
                    optimizer.state[new_param] = {
                        "step": stored_state.get("step", 0),
                        "exp_avg":    torch.zeros_like(new_logit),
                        "exp_avg_sq": torch.zeros_like(new_logit),
                    }
                group["params"][0] = new_param
                self._opacity = new_param
                break

    # ──────────────────────────────────────────────────────────────
    # Learning rate update
    # ──────────────────────────────────────────────────────────────

    def update_learning_rates(
        self,
        optimizer: torch.optim.Optimizer,
        iteration: int,
    ) -> None:
        """Apply exponential LR decay to position (xyz) parameter group."""
        for group in optimizer.param_groups:
            if "lr_init" in group:
                lr_fn = get_expon_lr_func(
                    lr_init=group["lr_init"],
                    lr_final=group["lr_final"],
                    lr_delay_mult=group.get("lr_delay_mult", 0.01),
                    max_steps=group["max_steps"],
                )
                group["lr"] = lr_fn(iteration)

    # ──────────────────────────────────────────────────────────────
    # PLY I/O (compatible with standard 3DGS viewers)
    # ──────────────────────────────────────────────────────────────

    def save_ply(self, path: str | Path) -> None:
        """
        Save model to PLY file.

        The PLY file stores all raw (pre-activation) parameter values:
        positions, SH DC, SH rest, logit-opacity, log-scale, raw rotation.
        This allows exact reconstruction without precision loss.

        Compatible with the original 3DGS viewer and gaussian-splatting-cuda.
        """
        from plyfile import PlyData, PlyElement

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        xyz      = self._xyz.detach().cpu().numpy().astype(np.float32)
        normals  = np.zeros_like(xyz)
        fdc      = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).cpu().numpy()
        frest    = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).cpu().numpy()
        opacities= self._opacity.detach().cpu().numpy()
        scale    = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        attrs = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        attrs += [f'f_dc_{i}'   for i in range(fdc.shape[1])]
        attrs += [f'f_rest_{i}' for i in range(frest.shape[1])]
        attrs += ['opacity']
        attrs += [f'scale_{i}'  for i in range(scale.shape[1])]
        attrs += [f'rot_{i}'    for i in range(rotation.shape[1])]

        dtype_full = [(a, 'f4') for a in attrs]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        all_props = np.concatenate([xyz, normals, fdc, frest, opacities, scale, rotation], axis=1)
        elements[:] = list(map(tuple, all_props))

        PlyData([PlyElement.describe(elements, 'vertex')]).write(str(path))
        print(f"[GaussianModel] Saved {xyz.shape[0]:,} Gaussians to {path}")

    def load_ply(self, path: str | Path) -> None:
        """Load Gaussian model from PLY file."""
        from plyfile import PlyData

        plydata = PlyData.read(str(path))
        verts = plydata['vertex']

        xyz = np.stack([verts['x'], verts['y'], verts['z']], axis=1)

        fdc_names  = sorted([p.name for p in verts.properties if p.name.startswith('f_dc_')])
        fdc = np.stack([verts[n] for n in fdc_names], axis=1).reshape(-1, 1, 3)

        frest_names = sorted([p.name for p in verts.properties if p.name.startswith('f_rest_')])
        K = num_sh_coefficients(self.max_sh_degree) - 1
        if frest_names:
            frest = np.stack([verts[n] for n in frest_names], axis=1).reshape(-1, K, 3)
        else:
            frest = np.zeros((xyz.shape[0], K, 3), dtype=np.float32)

        opacities = verts['opacity'].reshape(-1, 1).astype(np.float32)
        scale     = np.stack([verts[f'scale_{i}'] for i in range(3)], axis=1).astype(np.float32)
        rotation  = np.stack([verts[f'rot_{i}']   for i in range(4)], axis=1).astype(np.float32)

        self._xyz           = nn.Parameter(torch.tensor(xyz,       dtype=torch.float32))
        self._features_dc   = nn.Parameter(torch.tensor(fdc,       dtype=torch.float32))
        self._features_rest = nn.Parameter(torch.tensor(frest,     dtype=torch.float32))
        self._opacity       = nn.Parameter(torch.tensor(opacities, dtype=torch.float32))
        self._scaling       = nn.Parameter(torch.tensor(scale,     dtype=torch.float32))
        self._rotation      = nn.Parameter(torch.tensor(rotation,  dtype=torch.float32))

        self.active_sh_degree = self.max_sh_degree
        self._init_densification_stats()
        print(f"[GaussianModel] Loaded {xyz.shape[0]:,} Gaussians from {path}")

    def __repr__(self) -> str:
        n = self.num_gaussians
        return (f"GaussianModel(N={n:,}, sh_degree={self.sh_degree}, "
                f"active_sh={self.active_sh_degree})")
