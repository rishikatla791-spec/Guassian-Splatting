"""
Visibility culling utilities for efficient Gaussian rendering.

Frustum Culling Math
--------------------
Extract 6 frustum planes from the combined projection-view matrix M = P·V.

Given clip-space point p' = M·p (homogeneous), the frustum planes are:
  left:   M[3] + M[0] > 0
  right:  M[3] - M[0] > 0
  bottom: M[3] + M[1] > 0
  top:    M[3] - M[1] > 0
  near:   M[3] + M[2] > 0
  far:    M[3] - M[2] > 0

where M[i] is the i-th row of the full 4×4 matrix.

Each plane normal (A,B,C,D) satisfies: A·x + B·y + C·z + D > 0 for inside points.
"""

from __future__ import annotations
from typing import Tuple, List

import torch
import numpy as np


class VisibilityCuller:
    """
    Frustum culling and depth sorting for 3D Gaussian Splatting.
    """

    @staticmethod
    def extract_frustum_planes(proj_view_matrix: torch.Tensor) -> torch.Tensor:
        """
        Extract 6 frustum plane equations from combined P·V matrix.

        Args:
            proj_view_matrix: (4,4) combined projection×view matrix

        Returns:
            planes: (6, 4) each row is [A, B, C, D] for Ax+By+Cz+D > 0
        """
        M = proj_view_matrix
        planes = torch.stack([
            M[3] + M[0],   # left
            M[3] - M[0],   # right
            M[3] + M[1],   # bottom
            M[3] - M[1],   # top
            M[3] + M[2],   # near
            M[3] - M[2],   # far
        ], dim=0)  # (6, 4)

        # Normalize planes (unit normal)
        norms = planes[:, :3].norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return planes / norms

    @staticmethod
    def frustum_cull(
        points: torch.Tensor,
        viewmatrix: torch.Tensor,
        projmatrix: torch.Tensor,
        margin: float = 0.0,
    ) -> torch.Tensor:
        """
        Frustum-cull 3D points.

        A point is inside the frustum if it satisfies all 6 half-space inequalities.

        Args:
            points:     (N, 3) world-space points
            viewmatrix: (4,4) world-to-camera
            projmatrix: (4,4) camera-to-clip
            margin:     extra margin (in world units) added to frustum planes

        Returns:
            visible: (N,) bool tensor, True = inside frustum
        """
        device = points.device
        pv = projmatrix @ viewmatrix
        planes = VisibilityCuller.extract_frustum_planes(pv).to(device)  # (6,4)

        N = points.shape[0]
        ones = torch.ones(N, 1, device=device, dtype=points.dtype)
        ph = torch.cat([points, ones], dim=-1)  # (N, 4)

        # Distance from each point to each plane: (N, 6)
        dists = ph @ planes.T
        visible = (dists >= -margin).all(dim=-1)
        return visible

    @staticmethod
    def depth_sort(depths: torch.Tensor) -> torch.Tensor:
        """
        Sort indices by depth for front-to-back compositing.

        Args:
            depths: (N,) depth values (camera-space Z)

        Returns:
            (N,) sort indices, ascending (closest first)
        """
        return torch.argsort(depths, stable=True)

    @staticmethod
    def compute_tiles(
        means2d: torch.Tensor,
        radii: torch.Tensor,
        img_h: int,
        img_w: int,
        tile_size: int = 16,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute which tiles each Gaussian overlaps.

        For each Gaussian i with center (cx, cy) and radius r,
        it overlaps tile (tx, ty) iff:
          tile_x0 <= cx + r  and  cx - r <= tile_x1
          tile_y0 <= cy + r  and  cy - r <= tile_y1

        Returns:
            tile_x_min: (N,) first tile column index
            tile_x_max: (N,) last tile column index (inclusive)
            tile_y_min: (N,) first tile row index
            tile_y_max: (N,) last tile row index (inclusive)
        """
        n_tiles_x = (img_w + tile_size - 1) // tile_size
        n_tiles_y = (img_h + tile_size - 1) // tile_size

        r = radii.float()
        cx, cy = means2d[:, 0], means2d[:, 1]

        tx_min = ((cx - r) / tile_size).long().clamp(0, n_tiles_x - 1)
        tx_max = ((cx + r) / tile_size).long().clamp(0, n_tiles_x - 1)
        ty_min = ((cy - r) / tile_size).long().clamp(0, n_tiles_y - 1)
        ty_max = ((cy + r) / tile_size).long().clamp(0, n_tiles_y - 1)

        return tx_min, tx_max, ty_min, ty_max

    @staticmethod
    def count_tiles_per_gaussian(
        tx_min: torch.Tensor,
        tx_max: torch.Tensor,
        ty_min: torch.Tensor,
        ty_max: torch.Tensor,
    ) -> torch.Tensor:
        """
        Count number of tiles touched by each Gaussian.

        Returns:
            (N,) tile count per Gaussian
        """
        return (tx_max - tx_min + 1) * (ty_max - ty_min + 1)
