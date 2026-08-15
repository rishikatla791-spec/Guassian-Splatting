"""
GaussianTrainer: Full training loop for 3D Gaussian Splatting.

Training Schedule (Kerbl et al. 2023)
──────────────────────────────────────
Iter 1-500:       warm-up (no densification)
Iter 500-15k:     densify every 100 steps
Iter 1k, 2k, 3k: increase SH degree
Every 3k:         reset opacity
Iter 7k, 30k:     save checkpoints

Optimizer
─────────
Adam with per-parameter learning rates and eps=1e-15
(small eps improves numerical precision for small gradients).

Position LR decays exponentially:
  lr(t) = lr_init · (lr_final/lr_init)^(t/T)

All other LRs are constant.

Gradient Management
──────────────────
Gradient clipping for rotation quaternions prevents
numerical instability from large quaternion gradients.
All gradients are clipped to 1.0 before optimizer step.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
import torch.optim as optim

try:
    from gaussian.core.gaussians import GaussianModel, get_expon_lr_func
except ImportError:
    try:
        from core.gaussians import GaussianModel, get_expon_lr_func
    except ImportError:
        from ..core.gaussians import GaussianModel, get_expon_lr_func
from .config import TrainingConfig
from .loss import combined_loss, l1_loss, psnr as compute_psnr_fn


class GaussianTrainer:
    """
    Full 3DGS training loop manager.

    Handles:
      - Per-parameter Adam optimizer with exponential LR schedule
      - Progressive SH degree increase
      - Adaptive densification and pruning
      - Opacity resets for floater elimination
      - Checkpointing at configured iterations
      - Evaluation on test cameras
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.gaussians: Optional[GaussianModel] = None
        self.optimizer: Optional[optim.Adam] = None
        self.scene_cameras: List = []
        self.eval_cameras: List = []
        self._iteration: int = 0
        self._loss_history: List[float] = []
        self._psnr_history: List[float] = []

    # ─────────────────────────────────────────────────────────────
    # Setup
    # ─────────────────────────────────────────────────────────────

    def setup(
        self,
        gaussians: GaussianModel,
        scene_cameras: List,
        eval_cameras: List = [],
    ) -> None:
        """
        Initialize Adam optimizer with per-parameter learning rates.

        Parameter groups and learning rates:
          _xyz:           position_lr_init  (decays exponentially)
          _features_dc:   feature_lr
          _features_rest: feature_lr / 20.0  (slower: higher SH harder to train)
          _opacity:       opacity_lr
          _scaling:       scaling_lr
          _rotation:      rotation_lr

        Adam eps=1e-15 (vs default 1e-8) improves convergence for parameters
        with very small gradients (e.g., high-degree SH coefficients).
        """
        self.gaussians = gaussians
        self.scene_cameras = scene_cameras
        self.eval_cameras  = eval_cameras

        # Pre-cache image tensors and scale intrinsics once before training
        for cam in self.scene_cameras:
            if hasattr(cam, 'load_image'):
                cam.load_image()
        for cam in self.eval_cameras:
            if hasattr(cam, 'load_image'):
                cam.load_image()

        cfg = self.config
        param_groups = [
            {
                "name":          "_xyz",
                "params":        [gaussians._xyz],
                "lr":            cfg.position_lr_init,
                "lr_init":       cfg.position_lr_init,
                "lr_final":      cfg.position_lr_final,
                "lr_delay_mult": cfg.position_lr_delay_mult,
                "max_steps":     cfg.position_lr_max_steps,
            },
            {
                "name":   "_features_dc",
                "params": [gaussians._features_dc],
                "lr":     cfg.feature_lr,
            },
            {
                "name":   "_features_rest",
                "params": [gaussians._features_rest],
                "lr":     cfg.feature_lr / 20.0,
            },
            {
                "name":   "_opacity",
                "params": [gaussians._opacity],
                "lr":     cfg.opacity_lr,
            },
            {
                "name":   "_scaling",
                "params": [gaussians._scaling],
                "lr":     cfg.scaling_lr,
            },
            {
                "name":   "_rotation",
                "params": [gaussians._rotation],
                "lr":     cfg.rotation_lr,
            },
        ]

        self.optimizer = optim.Adam(param_groups, eps=1e-15)
        print(f"[GaussianTrainer] Optimizer initialized with {sum(p.numel() for g in param_groups for p in g['params']):,} parameters")

    # ─────────────────────────────────────────────────────────────
    # LR update
    # ─────────────────────────────────────────────────────────────

    def _update_lr(self, iteration: int) -> None:
        """Update exponential LR schedule for all parameter groups that have lr_init."""
        for group in self.optimizer.param_groups:
            if "lr_init" in group:
                lr_fn = get_expon_lr_func(
                    lr_init=group["lr_init"],
                    lr_final=group["lr_final"],
                    lr_delay_mult=group.get("lr_delay_mult", 0.01),
                    max_steps=group["max_steps"],
                )
                group["lr"] = lr_fn(iteration)

    # ─────────────────────────────────────────────────────────────
    # Single train step
    # ─────────────────────────────────────────────────────────────

    def train_step(
        self,
        camera,
        renderer,
        iteration: int,
    ) -> Dict[str, float]:
        """
        Execute one training iteration.

        Training order:
            1. Progressive SH degree update
            2. Forward pass (render)
            3. Load GT image, compute loss
            4. Backward pass
            5. Accumulate densification stats
            6. Densify/prune if scheduled
            7. Opacity reset if scheduled
            8. Optimizer step + LR update

        Args:
            camera:    Camera instance to render from
            renderer:  TileBasedRasterizer
            iteration: current iteration number (1-indexed)

        Returns:
            dict with 'loss', 'l1_loss', 'psnr' metrics
        """
        cfg       = self.config
        gaussians = self.gaussians

        # ── Progressive SH degree ───────────────────────────────
        # SH degree increases at fixed iterations, allowing the model
        # to first learn low-frequency color, then view-dependent detail
        sh_milestones = [1000, 1500, 2000, 2500, 3000]
        for milestone in sh_milestones:
            if iteration == milestone:
                gaussians.oneupSHdegree()

        # ── Background color ──────────────────────────────────
        device = gaussians.get_xyz.device
        if cfg.white_background:
            bg = torch.ones(3, device=device)
        elif getattr(cfg, 'random_background', False):
            bg = torch.rand(3, device=device) if iteration % 5 != 0 else torch.zeros(3, device=device)
        else:
            bg = torch.zeros(3, device=device)

        # ── Forward: render ────────────────────────────────────
        render_out = renderer.render(gaussians, camera, bg_color=bg)
        rendered          = render_out["render"]             # (3, H, W)
        viewspace_points  = render_out["viewspace_points"]
        visibility_filter = render_out["visibility_filter"]
        radii             = render_out["radii"]

        # ── Ground truth ──────────────────────────────────────
        gt_image = camera.load_image().to(device)
        if gt_image.shape[-2:] != rendered.shape[-2:]:
            import torch.nn.functional as F
            gt_image = F.interpolate(
                gt_image.unsqueeze(0),
                size=rendered.shape[-2:],
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)

        # ── Loss ───────────────────────────────────────────
        loss = combined_loss(rendered, gt_image, lambda_dssim=cfg.lambda_dssim)
        l1   = l1_loss(rendered, gt_image).item()
        psnr_val = compute_psnr_fn(rendered.detach(), gt_image)

        # ── Backward ─────────────────────────────────────────
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # ── Densification statistics ───────────────────────────
        densify_start = min(cfg.densify_from_iter, max(10, int(0.05 * cfg.iterations)))
        densify_end   = min(cfg.densify_until_iter, max(50, int(0.80 * cfg.iterations)))
        densify_freq  = max(5, min(cfg.densification_interval, max(5, int(0.02 * cfg.iterations))))

        if iteration < densify_end:
            # Track max 2D radius per Gaussian
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter],
                radii[visibility_filter].float()
            )
            # Accumulate 2D gradient norms
            gaussians.add_densification_stats(viewspace_points, visibility_filter)

            # ── Densify / prune ───────────────────────
            if (
                iteration > densify_start and
                iteration % densify_freq == 0
            ):
                scene_extent = self._estimate_scene_extent()
                gaussians.densify_and_prune(
                    max_grad=cfg.densify_grad_threshold,
                    min_opacity=cfg.min_opacity,
                    extent=scene_extent,
                    max_screen_size=cfg.max_screen_size,
                    optimizer=self.optimizer,
                )

            # ── Opacity reset ───────────────────────────
            if iteration % cfg.opacity_reset_interval == 0:
                gaussians.reset_opacity(self.optimizer)

        # ── Gradient clipping ───────────────────────────────
        # Clip rotation quaternion gradients to prevent numerical blow-up
        torch.nn.utils.clip_grad_norm_(
            [p for g in self.optimizer.param_groups if g.get('name') == '_rotation'
             for p in g['params']],
            max_norm=1.0,
        )

        # ── Optimizer step + LR update ─────────────────────
        self.optimizer.step()
        self._update_lr(iteration)

        # Track history
        self._loss_history.append(loss.item())
        self._psnr_history.append(psnr_val)

        return {
            "loss":    loss.item(),
            "l1_loss": l1,
            "psnr":    psnr_val,
        }

    def _estimate_scene_extent(self) -> float:
        """
        Estimate scene spatial extent from camera positions.
        Used to scale densification thresholds.
        Returns max camera-to-centroid distance.
        """
        if not self.scene_cameras:
            return 1.0
        centers = torch.stack([
            cam.camera_center for cam in self.scene_cameras
        ]).float()
        centroid = centers.mean(dim=0)
        dists = (centers - centroid).norm(dim=-1)
        return float(dists.max().item())

    # ─────────────────────────────────────────────────────────────
    # Full training loop
    # ─────────────────────────────────────────────────────────────

    def train(
        self,
        renderer,
        callbacks: List[Callable] = [],
        log_interval: int = 100,
        random_camera: bool = True,
    ) -> GaussianModel:
        """
        Run the full training loop.

        Camera selection strategy:
            random_camera=True (default):  random camera sampling provides
            better gradient diversity and prevents overfitting to ordering.
            random_camera=False: round-robin ordering.

        Args:
            renderer:      TileBasedRasterizer instance
            callbacks:     list of callables called with (iteration, metrics)
            log_interval:  print metrics every N iterations
            random_camera: use random camera sampling (recommended)

        Returns:
            trained GaussianModel
        """
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False

        cfg = self.config
        n_cameras = len(self.scene_cameras)
        if n_cameras == 0:
            raise ValueError("No training cameras provided!")

        t0 = time.time()
        iters = range(1, cfg.iterations + 1)
        if use_tqdm:
            iters = tqdm(iters, desc="Training 3DGS", dynamic_ncols=True)

        for iteration in iters:
            self._iteration = iteration

            # Camera selection
            if random_camera:
                cam_idx = torch.randint(0, n_cameras, (1,)).item()
            else:
                cam_idx = (iteration - 1) % n_cameras
            camera = self.scene_cameras[cam_idx]

            # Training step
            metrics = self.train_step(camera, renderer, iteration)

            # Logging
            if iteration % log_interval == 0 or iteration == 1:
                elapsed = time.time() - t0
                n_gauss = self.gaussians.num_gaussians
                lr_xyz  = next(
                    (g['lr'] for g in self.optimizer.param_groups if g.get('name') == '_xyz'),
                    0.0
                )
                if use_tqdm:
                    iters.set_postfix({
                        "loss":  f"{metrics['loss']:.4f}",
                        "PSNR":  f"{metrics['psnr']:.2f}dB",
                        "N":     f"{n_gauss:,}",
                        "lr":    f"{lr_xyz:.2e}",
                    })
                else:
                    print(
                        f"[{iteration:5d}/{cfg.iterations}] "
                        f"loss={metrics['loss']:.4f} "
                        f"PSNR={metrics['psnr']:.2f}dB "
                        f"N={n_gauss:,} "
                        f"lr={lr_xyz:.2e} "
                        f"t={elapsed:.0f}s"
                    )

            # Callbacks
            for cb in callbacks:
                cb(iteration, metrics)

            # Checkpoints
            if iteration in cfg.save_iterations:
                out_path = (
                    Path(cfg.model_path)
                    / "point_cloud"
                    / f"iteration_{iteration}"
                    / "point_cloud.ply"
                )
                self.gaussians.save_ply(out_path)

        print(f"Training complete. Final N={self.gaussians.num_gaussians:,}")
        return self.gaussians

    # ─────────────────────────────────────────────────────────────
    # Evaluation
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def compute_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
        """Compute PSNR in dB. Higher = better."""
        return compute_psnr_fn(pred, gt)

    # ─────────────────────────────────────────────────────────────
    # Checkpointing
    # ─────────────────────────────────────────────────────────────

    def save_checkpoint(self, path: str | Path) -> None:
        """Save full training state: model params + optimizer state + config."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "iteration":      self._iteration,
            "active_sh_degree": self.gaussians.active_sh_degree,
            "gaussians_state": {
                "_xyz":           self.gaussians._xyz.detach(),
                "_features_dc":   self.gaussians._features_dc.detach(),
                "_features_rest": self.gaussians._features_rest.detach(),
                "_opacity":       self.gaussians._opacity.detach(),
                "_scaling":       self.gaussians._scaling.detach(),
                "_rotation":      self.gaussians._rotation.detach(),
            },
            "optimizer_state": self.optimizer.state_dict(),
            "config":          self.config.to_dict(),
            "loss_history":    self._loss_history[-1000:],  # last 1000 losses
            "psnr_history":    self._psnr_history[-1000:],
        }, str(path))
        print(f"[Checkpoint] Saved iteration {self._iteration} to {path}")

    def load_checkpoint(self, path: str | Path) -> None:
        """Load training state from checkpoint."""
        ckpt = torch.load(str(path), map_location="cpu")
        self._iteration = ckpt["iteration"]
        self.gaussians.active_sh_degree = ckpt["active_sh_degree"]

        gs = self.gaussians
        for key, val in ckpt["gaussians_state"].items():
            setattr(gs, key, torch.nn.Parameter(val))
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self._loss_history = ckpt.get("loss_history", [])
        self._psnr_history = ckpt.get("psnr_history", [])
        print(f"[Checkpoint] Loaded iteration {self._iteration} from {path}")

    @property
    def loss_history(self) -> List[float]:
        return self._loss_history

    @property
    def psnr_history(self) -> List[float]:
        return self._psnr_history
