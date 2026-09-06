"""
GaussianModel: Core learnable 3D Gaussian scene representation.

Matches official Inria 3D Gaussian Splatting (gaussian-splatting-main):
  - Parametrization:
      μ  ∈ ℝ³        positions (_xyz)
      q  ∈ ℝ⁴        unit quaternions (_rotation)
      s  ∈ ℝ³        log-scales (_scaling)
      α  ∈ ℝ         logit-opacities (_opacity)
      f_dc ∈ ℝ^(1×3) SH DC base color (_features_dc)
      f_rest ∈ ℝ^(15×3) SH Rest higher-degree coefficients (_features_rest)
  - Full Adaptive Densification:
      - Clone: duplicate small under-reconstructed Gaussians (||∇_2D|| ≥ τ_pos, scale ≤ percent_dense · scene_extent)
      - Split: divide oversized Gaussians (||∇_2D|| ≥ τ_pos, scale > percent_dense · scene_extent) into N=2 sub-Gaussians
      - Prune: remove low-opacity (α < min_opacity) or oversized (2D radius > max_screen_size or 3D scale > 0.1 · extent)
      - Full Adam optimizer state synchronization (exp_avg, exp_avg_sq) during all topology mutations
  - Spherical Harmonics: Progressive activation from degree 0 → degree 3 (16 coefficients per channel = 48 total)
  - Periodic Opacity Reset: reset opacities to logit(0.01) every 3,000 steps to eradicate floaters
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree

from .sh import RGB2SH, num_sh_coefficients, eval_sh
from .math_utils import inverse_sigmoid, safe_log, quaternion_to_rotation_matrix, build_covariance_3d


# ══════════════════════════════════════════════════════════════════
# Exponential LR Schedule
# ══════════════════════════════════════════════════════════════════

def get_expon_lr_func(
    lr_init: float,
    lr_final: float,
    lr_delay_steps: int = 0,
    lr_delay_mult: float = 1.0,
    max_steps: int = 30_000,
):
    """
    Exponential learning rate decay schedule matching official 3DGS.

    lr(t) = lr_init · (lr_final / lr_init)^(t / max_steps)
    With optional sine warmup if lr_delay_steps > 0.
    """
    if lr_init == 0.0 and lr_final == 0.0:
        return lambda step: 0.0

    def func(step: int) -> float:
        if step < 0:
            return 0.0

        if lr_delay_steps > 0 and step < lr_delay_steps:
            delay_rate = lr_delay_mult + (1.0 - lr_delay_mult) * math.sin(
                0.5 * math.pi * float(step) / float(lr_delay_steps)
            )
        else:
            delay_rate = 1.0

        t = min(float(step) / max(float(max_steps), 1.0), 1.0)
        log_lerp = lr_init * ((lr_final / lr_init) ** t)
        return float(delay_rate * log_lerp)

    return func


# ══════════════════════════════════════════════════════════════════
# GaussianModel
# ══════════════════════════════════════════════════════════════════

class GaussianModel(nn.Module):
    """
    Learnable 3D Gaussian scene representation.
    """

    def __init__(self, sh_degree: int = 3):
        super().__init__()
        assert 0 <= sh_degree <= 3, f"sh_degree must be in [0,3], got {sh_degree}"

        self.max_sh_degree = sh_degree
        self.active_sh_degree = 0
        self.percent_dense = 0.01
        self.spatial_lr_scale = 1.0

        # Learnable parameters (nn.Parameter)
        self._xyz = nn.Parameter(torch.empty(0))
        self._features_dc = nn.Parameter(torch.empty(0))
        self._features_rest = nn.Parameter(torch.empty(0))
        self._scaling = nn.Parameter(torch.empty(0))
        self._rotation = nn.Parameter(torch.empty(0))
        self._opacity = nn.Parameter(torch.empty(0))

        # Densification tracking tensors (not parameters)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)

    # ──────────────────────────────────────────────────────────────
    # Properties & Activations
    # ──────────────────────────────────────────────────────────────

    @property
    def num_gaussians(self) -> int:
        return self._xyz.shape[0] if self._xyz.numel() > 0 else 0

    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz

    @property
    def get_features(self) -> torch.Tensor:
        """Returns full (N, K, 3) SH coefficients."""
        if self._features_rest.numel() == 0:
            return self._features_dc
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_features_dc(self) -> torch.Tensor:
        return self._features_dc

    @property
    def get_features_rest(self) -> torch.Tensor:
        return self._features_rest

    @property
    def get_scaling(self) -> torch.Tensor:
        return torch.exp(self._scaling)

    @property
    def get_rotation(self) -> torch.Tensor:
        return F.normalize(self._rotation, p=2, dim=-1)

    @property
    def get_opacity(self) -> torch.Tensor:
        return torch.sigmoid(self._opacity)

    def get_covariance(self, scale_modifier: float = 1.0) -> torch.Tensor:
        return build_covariance_3d(self._scaling, scale_modifier, self._rotation)

    # ──────────────────────────────────────────────────────────────
    # SH Degree Management
    # ──────────────────────────────────────────────────────────────

    def oneupSHdegree(self) -> None:
        """Increase active SH degree by 1 up to max_sh_degree."""
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            print(f"[GaussianModel] Activated SH Degree: {self.active_sh_degree} / {self.max_sh_degree}")

    # ──────────────────────────────────────────────────────────────
    # Initialization
    # ──────────────────────────────────────────────────────────────

    def init_from_pointcloud(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        spatial_lr_scale: float = 1.0,
        device: str = "cuda",
    ) -> None:
        """
        Initialize Gaussians from point cloud (e.g. COLMAP sparse points).
        """
        self.spatial_lr_scale = spatial_lr_scale
        N = points.shape[0]
        if not torch.cuda.is_available() or device == "cpu":
            device = "cpu"
        else:
            device = "cuda"

        pts_tensor = torch.tensor(points, dtype=torch.float32, device=device)

        # 1. Colors: RGB -> SH DC
        if colors is None:
            colors = np.full((N, 3), 0.5, dtype=np.float32)
        cols_tensor = torch.tensor(colors, dtype=torch.float32, device=device)
        f_dc = RGB2SH(cols_tensor).unsqueeze(1) # (N, 1, 3)

        # Extra SH features for degrees 1..max_sh_degree
        n_rest = num_sh_coefficients(self.max_sh_degree) - 1
        f_rest = torch.zeros((N, n_rest, 3), dtype=torch.float32, device=device)

        # 2. Scales: isotropic initialization using 3-NN mean distance
        if N > 3:
            tree = cKDTree(points)
            dists, _ = tree.query(points, k=4)
            mean_dist = np.mean(dists[:, 1:], axis=1) # Exclude self
            mean_dist = np.clip(mean_dist, 1e-7, None)
            init_scale = np.log(mean_dist)[:, None].repeat(3, axis=1)
        else:
            init_scale = np.full((N, 3), np.log(0.05), dtype=np.float32)

        scales_tensor = torch.tensor(init_scale, dtype=torch.float32, device=device)

        # 3. Rotations: identity quaternion [1, 0, 0, 0]
        rots_tensor = torch.zeros((N, 4), dtype=torch.float32, device=device)
        rots_tensor[:, 0] = 1.0

        # 4. Opacities: inverse_sigmoid(0.1) ≈ -2.1972
        opac_init = inverse_sigmoid(torch.full((N, 1), 0.1, dtype=torch.float32, device=device))

        # Register learnable nn.Parameters
        self._xyz = nn.Parameter(pts_tensor.requires_grad_(True))
        self._features_dc = nn.Parameter(f_dc.requires_grad_(True))
        self._features_rest = nn.Parameter(f_rest.requires_grad_(True))
        self._scaling = nn.Parameter(scales_tensor.requires_grad_(True))
        self._rotation = nn.Parameter(rots_tensor.requires_grad_(True))
        self._opacity = nn.Parameter(opac_init.requires_grad_(True))

        self.active_sh_degree = 0
        self._init_densification_stats()
        print(f"[GaussianModel] Initialized {N:,} Gaussians (max_sh={self.max_sh_degree}) on {device}")

    def _init_densification_stats(self) -> None:
        device = self._xyz.device if self._xyz.numel() > 0 else "cuda"
        N = self.num_gaussians
        self.xyz_gradient_accum = torch.zeros((N, 1), dtype=torch.float32, device=device)
        self.denom = torch.zeros((N, 1), dtype=torch.float32, device=device)
        self.max_radii2D = torch.zeros(N, dtype=torch.float32, device=device)

    # ──────────────────────────────────────────────────────────────
    # Densification Statistics Accumulation
    # ──────────────────────────────────────────────────────────────

    def add_densification_stats(
        self,
        viewspace_point_tensor: torch.Tensor,
        update_filter: torch.Tensor,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ) -> None:
        """
        Accumulate 2D NDC-space gradient norms on visible Gaussians,
        matching official Inria diff-gaussian-rasterization (backward.cu lines 527-528, 626-627).
        """
        if viewspace_point_tensor.grad is None:
            return
        # Gradient in screen space (x, y)
        grad_xy = viewspace_point_tensor.grad[update_filter, :2].clone()
        if image_width is not None and image_height is not None:
            grad_xy[:, 0] *= (0.5 * image_width)
            grad_xy[:, 1] *= (0.5 * image_height)
        self.xyz_gradient_accum[update_filter] += torch.norm(grad_xy, dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    # ──────────────────────────────────────────────────────────────
    # Optimizer State Synchronization
    # ──────────────────────────────────────────────────────────────

    def replace_tensor_to_optimizer(
        self,
        tensor: torch.Tensor,
        name: str,
        optimizer: torch.optim.Optimizer,
    ) -> nn.Parameter:
        """
        Replace parameter tensor in optimizer and reset its Adam moments.
        """
        for group in optimizer.param_groups:
            if group.get("name") == name:
                param = group["params"][0]
                stored_state = optimizer.state.get(param, None)
                new_param = nn.Parameter(tensor.requires_grad_(True))

                if stored_state is not None and "exp_avg" in stored_state:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                    del optimizer.state[param]
                    optimizer.state[new_param] = stored_state
                elif stored_state is not None:
                    del optimizer.state[param]

                group["params"][0] = new_param
                setattr(self, name, new_param)
                return new_param
        raise ValueError(f"Parameter group {name} not found in optimizer!")

    def _prune_optimizer(
        self,
        valid_mask: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """
        Slice optimizer parameters and Adam moments using valid_mask (True = keep).
        """
        for group in optimizer.param_groups:
            name = group.get("name")
            param = group["params"][0]
            stored_state = optimizer.state.get(param, None)

            new_data = param.data[valid_mask]
            new_param = nn.Parameter(new_data.requires_grad_(True))

            if stored_state is not None and "exp_avg" in stored_state:
                stored_state["exp_avg"] = stored_state["exp_avg"][valid_mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][valid_mask]
                del optimizer.state[param]
                optimizer.state[new_param] = stored_state
            elif stored_state is not None:
                del optimizer.state[param]

            group["params"][0] = new_param
            if name:
                setattr(self, name, new_param)

    def prune_points(
        self,
        mask: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """
        Remove points where mask is True.
        """
        valid_points_mask = ~mask
        if valid_points_mask.all():
            return

        self._prune_optimizer(valid_points_mask, optimizer)

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def _cat_tensors_to_optimizer(
        self,
        tensors_dict: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """
        Concatenate new parameter tensors and extend Adam moments with zeros.
        """
        for group in optimizer.param_groups:
            name = group.get("name")
            if name not in tensors_dict:
                continue

            param = group["params"][0]
            ext = tensors_dict[name]

            new_data = torch.cat([param.data, ext], dim=0)
            new_param = nn.Parameter(new_data.requires_grad_(True))

            stored_state = optimizer.state.get(param, None)
            if stored_state is not None and "exp_avg" in stored_state:
                zeros = torch.zeros_like(ext)
                stored_state["exp_avg"] = torch.cat([stored_state["exp_avg"], zeros], dim=0)
                stored_state["exp_avg_sq"] = torch.cat([stored_state["exp_avg_sq"], zeros], dim=0)
                del optimizer.state[param]
                optimizer.state[new_param] = stored_state
            elif stored_state is not None:
                del optimizer.state[param]

            group["params"][0] = new_param
            setattr(self, name, new_param)

    # ──────────────────────────────────────────────────────────────
    # Adaptive Densification: Clone & Split & Prune
    # ──────────────────────────────────────────────────────────────

    def densify_and_clone(
        self,
        grads: torch.Tensor,
        grad_threshold: float,
        scene_extent: float,
        optimizer: torch.optim.Optimizer,
    ) -> int:
        """
        Clone small under-reconstructed Gaussians with high 2D positional gradients.
        """
        # Selected if ||∇_2D|| >= threshold and max_scale <= percent_dense * scene_extent
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent
        )

        n_cloned = int(selected_pts_mask.sum().item())
        if n_cloned == 0:
            return 0

        new_xyz = self._xyz[selected_pts_mask].detach()
        new_features_dc = self._features_dc[selected_pts_mask].detach()
        new_features_rest = self._features_rest[selected_pts_mask].detach()
        new_opacities = self._opacity[selected_pts_mask].detach()
        new_scaling = self._scaling[selected_pts_mask].detach()
        new_rotation = self._rotation[selected_pts_mask].detach()

        self._cat_tensors_to_optimizer(
            {
                "_xyz": new_xyz,
                "_features_dc": new_features_dc,
                "_features_rest": new_features_rest,
                "_opacity": new_opacities,
                "_scaling": new_scaling,
                "_rotation": new_rotation,
            },
            optimizer,
        )

        device = self._xyz.device
        self.xyz_gradient_accum = torch.cat([self.xyz_gradient_accum, torch.zeros((n_cloned, 1), device=device)])
        self.denom = torch.cat([self.denom, torch.zeros((n_cloned, 1), device=device)])
        self.max_radii2D = torch.cat([self.max_radii2D, torch.zeros(n_cloned, device=device)])

        return n_cloned

    def densify_and_split(
        self,
        grads: torch.Tensor,
        grad_threshold: float,
        scene_extent: float,
        optimizer: torch.optim.Optimizer,
        N: int = 2,
    ) -> int:
        """
        Split large over-reconstructed Gaussians into N=2 smaller sub-Gaussians.
        """
        n_init_points = self.num_gaussians
        padded_grad = torch.zeros(n_init_points, device=self._xyz.device)
        padded_grad[:grads.shape[0]] = grads.squeeze()

        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent
        )

        n_selected = int(selected_pts_mask.sum().item())
        if n_selected == 0:
            return 0

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = quaternion_to_rotation_matrix(self._rotation[selected_pts_mask]).repeat(N, 1, 1)

        new_xyz = (rots @ samples.unsqueeze(-1)).squeeze(-1) + self._xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self._scaling[selected_pts_mask].repeat(N, 1) - math.log(0.8 * N)
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self._cat_tensors_to_optimizer(
            {
                "_xyz": new_xyz.detach(),
                "_features_dc": new_features_dc.detach(),
                "_features_rest": new_features_rest.detach(),
                "_opacity": new_opacity.detach(),
                "_scaling": new_scaling.detach(),
                "_rotation": new_rotation.detach(),
            },
            optimizer,
        )

        device = self._xyz.device
        self.xyz_gradient_accum = torch.cat([self.xyz_gradient_accum, torch.zeros((N * n_selected, 1), device=device)])
        self.denom = torch.cat([self.denom, torch.zeros((N * n_selected, 1), device=device)])
        self.max_radii2D = torch.cat([self.max_radii2D, torch.zeros(N * n_selected, device=device)])

        # Prune the original parent Gaussians (now replaced by the split copies)
        prune_filter = torch.cat([
            selected_pts_mask,
            torch.zeros(N * n_selected, dtype=torch.bool, device=device)
        ])
        self.prune_points(prune_filter, optimizer)

        return n_selected

    def densify_and_prune(
        self,
        max_grad: float,
        min_opacity: float,
        extent: float,
        max_screen_size: Optional[float],
        optimizer: torch.optim.Optimizer,
        radii: Optional[torch.Tensor] = None,
    ) -> Dict[str, int]:
        """
        Full adaptive densification step: Clone + Split + Prune.
        """
        grads = self.xyz_gradient_accum / self.denom.clamp(min=1.0)
        grads[grads.isnan()] = 0.0

        n_cloned = self.densify_and_clone(grads, max_grad, extent, optimizer)
        n_split = self.densify_and_split(grads, max_grad, extent, optimizer)

        # Prune low opacity
        prune_mask = (self.get_opacity < min_opacity).squeeze()

        # Prune oversized Gaussians
        if max_screen_size is not None and max_screen_size > 0:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > (0.1 * extent)
            prune_mask = prune_mask | big_points_vs | big_points_ws

        n_pruned = int(prune_mask.sum().item())
        self.prune_points(prune_mask, optimizer)

        # Reset statistics for next interval
        self.xyz_gradient_accum.zero_()
        self.denom.zero_()
        self.max_radii2D.zero_()
        torch.cuda.empty_cache()

        return {"n_cloned": n_cloned, "n_split": n_split, "n_pruned": n_pruned, "total": self.num_gaussians}

    # ──────────────────────────────────────────────────────────────
    # Periodic Opacity Reset
    # ──────────────────────────────────────────────────────────────

    def reset_opacity(
        self,
        optimizer: torch.optim.Optimizer,
        reset_value: float = 0.01,
    ) -> None:
        """
        Reset all opacities to min(current_opacity, reset_value) and zero Adam moments.
        """
        current_opacity = self.get_opacity
        clipped_opacity = torch.min(
            current_opacity,
            torch.full_like(current_opacity, reset_value)
        )
        new_logit = inverse_sigmoid(clipped_opacity)
        self.replace_tensor_to_optimizer(new_logit, "_opacity", optimizer)
        print(f"[GaussianModel] Opacity reset complete (capped at {reset_value}).")

    # ──────────────────────────────────────────────────────────────
    # PLY I/O
    # ──────────────────────────────────────────────────────────────

    def construct_list_of_attributes(self) -> List[str]:
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append(f'f_dc_{i}')
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append(f'f_rest_{i}')
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append(f'scale_{i}')
        for i in range(self._rotation.shape[1]):
            l.append(f'rot_{i}')
        return l

    def save_ply(self, path: str | Path) -> None:
        """Save raw parameters to standard 3DGS PLY format."""
        from plyfile import PlyData, PlyElement

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy().astype(np.float32)
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attr, 'f4') for attr in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(str(path))
        print(f"[GaussianModel] Saved {xyz.shape[0]:,} Gaussians to {path}")

    def load_ply(self, path: str | Path, device: str = "cuda") -> None:
        """Load Gaussians from standard 3DGS PLY format."""
        from plyfile import PlyData

        if not torch.cuda.is_available() or device == "cpu":
            device = "cpu"
        else:
            device = "cuda"

        plydata = PlyData.read(str(path))
        verts = plydata['vertex']

        xyz = np.stack([verts['x'], verts['y'], verts['z']], axis=1)
        opacities = verts['opacity'].reshape(-1, 1).astype(np.float32)

        f_dc_names = sorted([p.name for p in verts.properties if p.name.startswith('f_dc_')])
        f_dc = np.stack([verts[n] for n in f_dc_names], axis=1).reshape(-1, 3, 1).transpose(0, 2, 1)

        f_rest_names = sorted([p.name for p in verts.properties if p.name.startswith('f_rest_')])
        n_rest = len(f_rest_names) // 3
        if n_rest > 0:
            f_rest = np.stack([verts[n] for n in f_rest_names], axis=1).reshape(-1, 3, n_rest).transpose(0, 2, 1)
        else:
            n_req = num_sh_coefficients(self.max_sh_degree) - 1
            f_rest = np.zeros((xyz.shape[0], n_req, 3), dtype=np.float32)

        scale = np.stack([verts[f'scale_{i}'] for i in range(3)], axis=1).astype(np.float32)
        rotation = np.stack([verts[f'rot_{i}'] for i in range(4)], axis=1).astype(np.float32)

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float32, device=device).requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(f_dc, dtype=torch.float32, device=device).requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(f_rest, dtype=torch.float32, device=device).requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float32, device=device).requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scale, dtype=torch.float32, device=device).requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rotation, dtype=torch.float32, device=device).requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
        self._init_densification_stats()
        print(f"[GaussianModel] Loaded {xyz.shape[0]:,} Gaussians from {path}")

    def __repr__(self) -> str:
        return f"GaussianModel(N={self.num_gaussians:,}, max_sh={self.max_sh_degree}, active_sh={self.active_sh_degree})"
