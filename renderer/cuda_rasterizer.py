"""
CUDAGaussianRasterizer — High-Performance Differentiable CUDA Tile Rasterizer.

Hardware-accelerated CUDA implementation matching official Inria 3D Gaussian Splatting:
  - 3D Covariance Projection Σ = R S Sᵀ Rᵀ
  - EWA 2D Covariance Projection Σ' = (J W Σ Wᵀ Jᵀ)[:2,:2] + 0.3 I₂
  - Frustum Culling & 3σ Bounding Radius Filtering
  - View-dependent Spherical Harmonics Evaluation (Degrees 0..3)
  - Tile-based 16x16 CUDA alpha compositing with depth sorting
  - Full end-to-end differentiable backpropagation
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .gaussian_rasterizer import RasterizationSettings
    from ..core.cuda_ops import (
        build_covariance_3d_cuda,
        build_covariance_2d_cuda,
        invert_cov2d_cuda,
        compute_radius_cuda,
    )
    from ..core.cuda_rasterizer import cuda_rasterize
    from ..core.sh import eval_sh
    from ..core.math_utils import project_points, ndc_to_screen
except (ImportError, ValueError):
    from renderer.gaussian_rasterizer import RasterizationSettings
    from core.cuda_ops import (
        build_covariance_3d_cuda,
        build_covariance_2d_cuda,
        invert_cov2d_cuda,
        compute_radius_cuda,
    )
    from core.cuda_rasterizer import cuda_rasterize
    from core.sh import eval_sh
    from core.math_utils import project_points, ndc_to_screen


class CUDAGaussianRasterizer(nn.Module):
    """
    Differentiable C++/CUDA Gaussian Rasterizer for 3D Gaussian Splatting.
    """

    def __init__(self, raster_settings: RasterizationSettings) -> None:
        super().__init__()
        self.raster_settings = raster_settings

    def forward(
        self,
        means3d: torch.Tensor,                  # (N, 3)
        means2d: torch.Tensor,                  # (N, 3) leaf proxy for gradient accum
        sh: Optional[torch.Tensor],             # (N, K, 3)
        colors_precomp: Optional[torch.Tensor], # (N, 3)
        opacities: torch.Tensor,                # (N, 1) or (N,)
        scales: Optional[torch.Tensor],         # (N, 3)
        rotations: Optional[torch.Tensor],      # (N, 4)
        cov3d_precomp: Optional[torch.Tensor],  # (N, 6)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        CUDA-Accelerated rendering pipeline.

        Returns:
            rendered_image: (H, W, 3) float32 in [0, 1]
            depth_map:      (H, W) float32
            radii:          (N,) int64 bounding radii
        """
        cfg = self.raster_settings
        N = means3d.shape[0]
        H, W = cfg.image_height, cfg.image_width

        device = means3d.device
        if device.type != "cuda":
            if torch.cuda.is_available():
                target_device = torch.device("cuda")
                means3d = means3d.to(target_device)
                means2d = means2d.to(target_device)
                if sh is not None: sh = sh.to(target_device)
                if colors_precomp is not None: colors_precomp = colors_precomp.to(target_device)
                opacities = opacities.to(target_device)
                if scales is not None: scales = scales.to(target_device)
                if rotations is not None: rotations = rotations.to(target_device)
                if cov3d_precomp is not None: cov3d_precomp = cov3d_precomp.to(target_device)
                device = target_device

        viewmatrix = cfg.viewmatrix.to(device)
        projmatrix = cfg.projmatrix.to(device)
        campos = cfg.campos.to(device)
        bg = cfg.bg.to(device)

        if N == 0:
            rendered = bg[None, None, :].expand(H, W, 3)
            depth_map = torch.zeros((H, W), device=device, dtype=torch.float32)
            radii = torch.zeros(0, device=device, dtype=torch.int64)
            return rendered, depth_map, radii

        # ── 1. 3D Covariance ────────────────────────────────────
        if cov3d_precomp is not None:
            covs3d = cov3d_precomp
        else:
            if scales is None or rotations is None:
                raise ValueError("Provide cov3d_precomp or (scales, rotations).")
            covs3d = build_covariance_3d_cuda(scales, cfg.scale_modifier, rotations)

        # ── 2. EWA 2D Projection ─────────────────────────────
        fovx = 2.0 * math.atan(cfg.tanfovx)
        fovy = 2.0 * math.atan(cfg.tanfovy)

        cov2d, t_cam = build_covariance_2d_cuda(
            means3d, covs3d, viewmatrix, fovx, fovy, W, H
        )

        # ── 3. Screen-space projection ─────────────────────────
        ndc, depths = project_points(means3d, viewmatrix, projmatrix)
        means2d_screen = ndc_to_screen(ndc, W, H)  # (N, 2)

        # Connect to leaf means2d proxy for densification gradient tracking
        # Both means3d (via means2d_screen) and means2d (viewspace_points) receive the exact screen-space gradient
        means2d_connected = means2d_screen + (means2d[:, :2] - means2d[:, :2].detach())

        # ── 4. GPU Frustum culling & bounding radius ─────────────────────
        if not cfg.prefiltered:
            visible = (
                (depths > 0.001) &
                (ndc[:, 0].abs() < 1.3) &
                (ndc[:, 1].abs() < 1.3)
            )
        else:
            visible = torch.ones(N, dtype=torch.bool, device=device)

        radii = compute_radius_cuda(cov2d, threshold=3.0)
        visible = visible & (radii > 0)

        # ── 5. View-dependent colors on GPU ─────────────────────────
        if colors_precomp is not None:
            colors = colors_precomp
        else:
            dirs = means3d - campos
            dirs = F.normalize(dirs, p=2, dim=-1)
            colors = eval_sh(cfg.sh_degree, sh, dirs)
            colors = (colors + 0.5).clamp(min=0.0)

        # ── 6. Invert 2D covariances analytically on GPU ─────────────
        cov2d_inv_mat, _ = invert_cov2d_cuda(cov2d)  # (N, 2, 2)
        # Extract [inv_00, inv_01, inv_11] as (N, 3)
        cov2d_inv = torch.stack([
            cov2d_inv_mat[:, 0, 0],
            cov2d_inv_mat[:, 0, 1],
            cov2d_inv_mat[:, 1, 1],
        ], dim=-1)

        # Flatten opacities to (N,)
        opac_1d = opacities.squeeze(-1) if opacities.dim() > 1 else opacities

        # ── 7. Differentiable CUDA Tile Rasterization ────────────────
        out_color, out_depth, out_alpha = cuda_rasterize(
            means2d=means2d_connected,
            colors=colors,
            opacities=opac_1d,
            cov2d_inv=cov2d_inv,
            depths=depths,
            radii=radii.float(),
            H=H,
            W=W,
            bg_color=bg,
            tile_size=16,
        )

        return out_color, out_depth, radii
