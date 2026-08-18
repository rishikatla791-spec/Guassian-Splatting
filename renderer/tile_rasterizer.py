"""
TileBasedRasterizer — high-level interface to GaussianRasterizer.

This class wraps the lower-level GaussianRasterizer and provides:
  - Convenient preprocessing of a GaussianModel + Camera pair.
  - A single `render()` call that returns a rich output dict.
  - Depth-sorted Gaussian indices for correct front-to-back compositing.

Coordinate Conventions
-----------------------
Camera space  : right-handed, z into scene (OpenCV)
Screen space  : x right, y down, origin at top-left, half-pixel offset
NDC           : [-1,1]³, y up (OpenGL)
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from ..core.math_utils import (
        build_covariance_2d,
        build_covariance_3d,
        compute_radius_from_cov2d,
        invert_cov2d,
        project_points,
        ndc_to_screen,
    )
    from ..core.sh import eval_sh
    from .gaussian_rasterizer import GaussianRasterizer, RasterizationSettings
    from .cuda_rasterizer import CUDAGaussianRasterizer
except (ImportError, ValueError):
    from core.math_utils import (
        build_covariance_2d,
        build_covariance_3d,
        compute_radius_from_cov2d,
        invert_cov2d,
        project_points,
        ndc_to_screen,
    )
    from core.sh import eval_sh
    from renderer.gaussian_rasterizer import GaussianRasterizer, RasterizationSettings
    from renderer.cuda_rasterizer import CUDAGaussianRasterizer


class TileBasedRasterizer:
    """
    High-level tile-based rasterizer for 3D Gaussian Splatting.

    TILE_SIZE specifies the pixel dimensions of each processing tile.
    Tiles are always square.

    Example
    -------
    ::

        rasterizer = TileBasedRasterizer()
        output = rasterizer.render(gaussians, camera, bg_color)
        rendered_image = output['render']   # (3, H, W)
    """

    TILE_SIZE: int = 16

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess_gaussians(
        self,
        gaussians,          # GaussianModel instance
        camera,             # Camera instance
        scale_modifier: float = 1.0,
        sh_degree: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Project 3D Gaussians to 2D and compute all per-Gaussian quantities
        needed for rasterization.

        Parameters
        ----------
        gaussians : GaussianModel
            Scene representation.
        camera : Camera
            The viewpoint from which to render.
        scale_modifier : float
            Global scale multiplier (default 1.0).
        sh_degree : int or None
            Active SH degree override.  Defaults to gaussians.active_sh_degree.

        Returns
        -------
        dict with keys:
          'means3d'      : (N, 3)   world-space centres
          'means2d'      : (N, 2)   screen-space projected centres (pixel coords)
          'cov2d'        : (N, 3)   2D covariance [a, b, c]
          'cov2d_inv'    : (N, 2, 2) inverse 2D covariance
          'colors'       : (N, 3)   view-dependent RGB
          'opacities'    : (N,)     sigmoid-activated opacities
          'depths'       : (N,)     camera-space depths
          'radii'        : (N,)     bounding radius in pixels (int)
          'ndc'          : (N, 3)   NDC coordinates
          'visible'      : (N,)     bool visibility mask
        """
        device = gaussians.get_xyz.device
        deg    = sh_degree if sh_degree is not None else gaussians.active_sh_degree

        # ---- Raw parameters ------------------------------------------------
        means3d   = gaussians.get_xyz              # (N, 3)
        opacities = gaussians.get_opacity.squeeze(-1)   # (N,)
        scales    = gaussians._scaling             # (N, 3) log-scale
        rotations = gaussians.get_rotation         # (N, 4) unit quat
        sh        = gaussians.get_features         # (N, K, 3)
        N = means3d.shape[0]

        # ---- Matrices to device --------------------------------------------
        viewmatrix = camera.view_matrix.to(device)                       # (4,4)
        projmatrix = camera.projection_matrix().to(device)               # (4,4)
        campos     = camera.camera_center.to(device)                     # (3,)

        fovx = camera.fovx   # radians
        fovy = camera.fovy
        W    = camera.width
        H    = camera.height

        # ---- 3D covariance -------------------------------------------------
        covs3d = build_covariance_3d(scales, scale_modifier, rotations)  # (N, 6)

        # ---- 2D projection -------------------------------------------------
        cov2d, t_cam = build_covariance_2d(
            means3d, covs3d, viewmatrix, fovx, fovy, W, H
        )   # cov2d: (N,3),  t_cam: (N,3) camera-space

        # ---- Screen-space projection ----------------------------------------
        ndc, depths = project_points(means3d, viewmatrix, projmatrix)   # ndc:(N,3), depths:(N,)
        means2d     = ndc_to_screen(ndc, W, H)                          # (N, 2)

        # ---- Visibility ----------------------------------------------------
        visible = (
            (depths > 0.0) &
            (ndc[:, 0].abs() < 1.3) &
            (ndc[:, 1].abs() < 1.3)
        )

        # ---- Bounding radius -----------------------------------------------
        radii = compute_radius_from_cov2d(cov2d, threshold=3.0)   # (N,) int64
        visible = visible & (radii > 0)

        # ---- Inverse covariance --------------------------------------------
        cov2d_inv, _ = invert_cov2d(cov2d)   # (N, 2, 2)

        # ---- View-dependent colour -----------------------------------------
        dirs   = F.normalize(means3d - campos[None, :], p=2, dim=-1)  # (N, 3)
        colors = eval_sh(deg, sh, dirs)     # (N, 3)
        colors = (colors + 0.5).clamp(min=0.0)

        return {
            "means3d":   means3d,
            "means2d":   means2d,
            "cov2d":     cov2d,
            "cov2d_inv": cov2d_inv,
            "colors":    colors,
            "opacities": opacities,
            "depths":    depths,
            "radii":     radii,
            "ndc":       ndc,
            "visible":   visible,
        }

    # ------------------------------------------------------------------
    # Depth sort
    # ------------------------------------------------------------------

    @staticmethod
    def sort_by_depth(
        depths: torch.Tensor,
        radii: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return indices that sort Gaussians from front to back.

        Invisible Gaussians (radii == 0) are placed last so they are never
        composited (their depth entry is irrelevant).

        Parameters
        ----------
        depths : (N,)  camera-space depth per Gaussian
        radii  : (N,)  bounding radius; 0 marks invisible Gaussians

        Returns
        -------
        sorted_indices : (N,)  int64 indices into the original Gaussian array
        """
        # Give invisible Gaussians a very large depth so they sort last
        effective_depths = depths.clone()
        effective_depths[radii == 0] = 1e10
        return torch.argsort(effective_depths)

    # ------------------------------------------------------------------
    # Full render
    # ------------------------------------------------------------------

    def render(
        self,
        gaussians,
        camera,
        bg_color: torch.Tensor,
        scale_modifier: float = 1.0,
        sh_degree: Optional[int] = None,
        override_colors: Optional[torch.Tensor] = None,
        compute_alpha: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Render a full frame.

        Parameters
        ----------
        gaussians : GaussianModel
        camera    : Camera
        bg_color  : (3,) background colour tensor
        scale_modifier : float
            Global scale multiplier.
        sh_degree : int or None
            Active SH degree.  Defaults to gaussians.active_sh_degree.
        override_colors : (N, 3) or None
            If provided, skip SH evaluation and use these colours directly.

        Returns
        -------
        dict with keys:
          'render'            : (3, H, W)  float32 rendered image
          'depth'             : (H, W)     float32 depth map
          'alpha'             : (H, W)     float32 accumulated alpha
          'viewspace_points'  : (N, 3)     screen-space proxy (requires_grad)
          'visibility_filter' : (N,)       bool, visible Gaussians
          'radii'             : (N,)       int bounding radii in pixels
        """
        device = gaussians.get_xyz.device
        deg    = sh_degree if sh_degree is not None else gaussians.active_sh_degree

        # ---- Pre-process ---------------------------------------------------
        pp = self.preprocess_gaussians(
            gaussians, camera,
            scale_modifier=scale_modifier,
            sh_degree=deg,
        )

        means3d   = pp["means3d"]
        means2d   = pp["means2d"]
        cov2d_inv = pp["cov2d_inv"]
        colors    = override_colors if override_colors is not None else pp["colors"]
        opacities = pp["opacities"]
        depths    = pp["depths"]
        radii     = pp["radii"]
        visible   = pp["visible"]

        H = camera.height
        W = camera.width

        # ---- Screen-space proxy for gradient accumulation ------------------
        # We create a leaf tensor from means2d that carries the graph.
        # Its xy-gradient magnitude is the densification signal.
        viewspace_points = torch.zeros(
            means3d.shape[0], 3,
            device=device,
            dtype=means3d.dtype,
            requires_grad=True,
        )
        # Attach to means2d computation graph via a dummy addition
        # (the true gradient flows back through the rasterizer's compositing)
        means2d_with_grad = means2d + 0.0 * viewspace_points[:, :2]

        # ---- Build RasterizationSettings -----------------------------------
        settings = RasterizationSettings(
            image_height=H,
            image_width=W,
            tanfovx=math.tan(camera.fovx / 2.0),
            tanfovy=math.tan(camera.fovy / 2.0),
            bg=bg_color.to(device),
            scale_modifier=scale_modifier,
            viewmatrix=camera.view_matrix.to(device),
            projmatrix=camera.projection_matrix().to(device),
            sh_degree=deg,
            campos=camera.camera_center.to(device),
            prefiltered=False,
            debug=False,
        )

        # ---- GaussianRasterizer call (CUDA vs PyTorch Vectorized) ------------
        try:
            from core.cuda_rasterizer import HAS_CUPY
        except (ImportError, ValueError):
            HAS_CUPY = False

        if device.type == "cuda" and HAS_CUPY:
            rasterizer = CUDAGaussianRasterizer(settings)
        else:
            rasterizer = GaussianRasterizer(settings)

        # Retrieve raw parameters for the rasterizer
        covs3d = gaussians.get_covariance(scale_modifier)
        sh_feats = gaussians.get_features

        rendered_hwc, depth_map, radii_out = rasterizer(
            means3d=means3d,
            means2d=viewspace_points,        # proxy leaf for grad tracking
            sh=sh_feats,
            colors_precomp=override_colors,
            opacities=gaussians.get_opacity,
            scales=gaussians._scaling,
            rotations=gaussians.get_rotation,
            cov3d_precomp=covs3d,
        )

        # rendered_hwc: (H, W, 3) → convert to (3, H, W)
        rendered_chw = rendered_hwc.permute(2, 0, 1)

        # ---- Accumulated alpha map -----------------------------------------
        # Alpha = 1 - T_final; we approximate from the rendered image vs bg
        # A more precise value would require storing T_buf, but for typical
        # use (background subtraction) this is equivalent when bg != rendered.
        # We compute it directly by re-running the transmittance: instead,
        # we return a proper alpha by computing it as 1 - T via accumulated
        # channel sum approach.  Since the rasterizer already composites bg,
        # alpha_map ≈ (rendered - bg) / (fg - bg).
        # A cleaner approach: alpha = 1 - T_final stored separately.
        # We compute it here as the sum of alpha weights.
        if compute_alpha:
            alpha_map = _compute_alpha_map(
                means2d=means2d[visible],
                opacities=opacities[visible],
                cov2d_inv=cov2d_inv[visible],
                depths=depths[visible],
                radii=radii[visible].float(),
                H=H,
                W=W,
                tile_size=self.TILE_SIZE,
            )
        else:
            alpha_map = torch.ones((H, W), device=device, dtype=means3d.dtype)

        return {
            "render":            rendered_chw,
            "depth":             depth_map,
            "alpha":             alpha_map,
            "viewspace_points":  viewspace_points,
            "visibility_filter": visible,
            "radii":             radii_out,
        }


# ---------------------------------------------------------------------------
# Alpha-map helper
# ---------------------------------------------------------------------------

def _compute_alpha_map(
    means2d: torch.Tensor,    # (M, 2)
    opacities: torch.Tensor,  # (M,)
    cov2d_inv: torch.Tensor,  # (M, 2, 2)
    depths: torch.Tensor,     # (M,)
    radii: torch.Tensor,      # (M,) float
    H: int,
    W: int,
    tile_size: int,
) -> torch.Tensor:
    """
    Compute accumulated alpha map (1 - T_final) using the same front-to-back
    compositing as the main rasterizer, but only tracking transmittance.

    Returns
    -------
    alpha_map : (H, W)  float32 in [0, 1]
    """
    device = means2d.device
    dtype  = means2d.dtype
    M = means2d.shape[0]
    if M == 0:
        return torch.zeros(H, W, device=device, dtype=dtype)

    # Sort front-to-back
    order      = torch.argsort(depths)
    s_means    = means2d[order]
    s_opa      = opacities[order]
    s_covinv   = cov2d_inv[order]
    s_depths   = depths[order]
    s_radii    = radii[order]

    n_tiles_y = math.ceil(H / tile_size)
    n_tiles_x = math.ceil(W / tile_size)

    T_buf = torch.ones(H, W, device=device, dtype=dtype)

    aabb_x_min = (s_means[:, 0] - s_radii).clamp(min=0)
    aabb_x_max = (s_means[:, 0] + s_radii).clamp(max=W - 1)
    aabb_y_min = (s_means[:, 1] - s_radii).clamp(min=0)
    aabb_y_max = (s_means[:, 1] + s_radii).clamp(max=H - 1)

    tile_x_min = (aabb_x_min / tile_size).long()
    tile_x_max = (aabb_x_max / tile_size).long().clamp(max=n_tiles_x - 1)
    tile_y_min = (aabb_y_min / tile_size).long()
    tile_y_max = (aabb_y_max / tile_size).long().clamp(max=n_tiles_y - 1)

    T_threshold     = 1e-4
    alpha_threshold = 1.0 / 255.0

    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            py0 = ty * tile_size
            px0 = tx * tile_size
            py1 = min(py0 + tile_size, H)
            px1 = min(px0 + tile_size, W)
            ph, pw = py1 - py0, px1 - px0

            tile_mask = (
                (tile_x_min <= tx) & (tx <= tile_x_max) &
                (tile_y_min <= ty) & (ty <= tile_y_max)
            )
            g_idx = tile_mask.nonzero(as_tuple=False).squeeze(-1)
            if g_idx.numel() == 0:
                continue

            py_c = torch.arange(py0, py1, device=device, dtype=dtype) + 0.5
            px_c = torch.arange(px0, px1, device=device, dtype=dtype) + 0.5
            grid_y, grid_x = torch.meshgrid(py_c, px_c, indexing="ij")
            pix = torch.stack([grid_x, grid_y], dim=-1)   # (ph, pw, 2)

            g_means  = s_means[g_idx]
            g_opa    = s_opa[g_idx]
            g_covinv = s_covinv[g_idx]

            T_tile = T_buf[py0:py1, px0:px1].clone()

            for gi in range(g_idx.numel()):
                mu   = g_means[gi]
                opa  = g_opa[gi]
                Cinv = g_covinv[gi]

                d     = pix - mu[None, None, :]
                d_row = d.unsqueeze(-2)
                Cinv_b = Cinv.unsqueeze(0).unsqueeze(0).expand(ph, pw, 2, 2)
                maha2  = (d_row @ Cinv_b @ d.unsqueeze(-1)).squeeze(-1).squeeze(-1)
                gauss_w   = torch.exp(-0.5 * maha2.clamp(max=20.0))
                eff_alpha = (opa * gauss_w).clamp(min=0.0, max=1.0 - 1e-5)

                T_tile = T_tile * (1.0 - eff_alpha)
                if T_tile.max().item() < T_threshold:
                    break

            T_buf[py0:py1, px0:px1] = T_tile

    return 1.0 - T_buf   # alpha = 1 - T_final
