"""
Dense Geometry Triangulation & High-Resolution Appearance Alignment Module.

Reconstructs dense geometry, performs statistical outlier filtering,
estimates surface normals, computes KNN distance scales, and prepares
high-fidelity initial parameters for 3D Gaussian Splatting.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import cv2
from scipy.spatial import cKDTree


class DenseGeometryReconstructor:
    """
    Multi-view dense point cloud geometry reconstructor.
    """

    def __init__(self, k_neighbors: int = 4, outlier_std_ratio: float = 2.0):
        self.k_neighbors = k_neighbors
        self.outlier_std_ratio = outlier_std_ratio

    def process_point_cloud(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        cameras: Optional[List] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Clean, densify, and compute geometric properties for point cloud.

        Args:
            points: (N, 3) float32 coordinates
            colors: (N, 3) float32 RGB values in [0, 1]
            cameras: Optional list of Camera objects for visual hull seeding

        Returns:
            clean_points: (M, 3)
            clean_colors: (M, 3)
            normals: (M, 3) estimated surface normals
            knn_dist: (M,) mean distance to k-nearest neighbors
        """
        print(f"=== [Dense Geometry] Processing point cloud: {len(points)} initial SIFT points ===")

        # 1. Multi-View Visual Hull Volumetric Seeding (fills textureless holes)
        dense_pts, dense_cols = self._seed_visual_hull(points, colors, cameras)

        # 2. Statistical Outlier Filter
        clean_pts, clean_cols = self._remove_statistical_outliers(dense_pts, dense_cols)
        print(f"[OK] Outlier filter: retained {len(clean_pts)} / {len(dense_pts)} points")

        # 3. KNN Distance Scale Computation
        knn_dist = self._compute_knn_distances(clean_pts)

        # 4. Surface Normal Estimation
        normals = self._estimate_normals(clean_pts)

        return clean_pts, clean_cols, normals, knn_dist

    def _seed_visual_hull(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        cameras: Optional[List] = None,
        grid_resolution: int = 18,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Multi-View Visual Hull Volumetric Seeding to fill textureless surface gaps."""
        if cameras is None or len(cameras) < 2 or len(points) == 0:
            return points, colors

        print(f"=== [Dense Geometry] Computing Multi-View Visual Hull Volumetric Seeding ===")
        min_p = np.min(points, axis=0) - 0.05
        max_p = np.max(points, axis=0) + 0.05

        xs = np.linspace(min_p[0], max_p[0], grid_resolution)
        ys = np.linspace(min_p[1], max_p[1], grid_resolution)
        zs = np.linspace(min_p[2], max_p[2], grid_resolution)
        grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing="ij")
        grid_pts = np.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], axis=1).astype(np.float32)

        valid_hull_pts = []
        valid_hull_cols = []
        cams = cameras[::max(1, len(cameras) // 10)]

        for p3d in grid_pts:
            hits = 0
            for cam in cams:
                R = cam.extrinsics.R
                T = cam.extrinsics.T
                p_cam = R @ p3d + T
                if p_cam[2] <= 0.1:
                    continue

                u = int(round(cam.intrinsics.fx * (p_cam[0] / p_cam[2]) + cam.intrinsics.cx))
                v = int(round(cam.intrinsics.fy * (p_cam[1] / p_cam[2]) + cam.intrinsics.cy))

                if 0 <= u < cam.intrinsics.width and 0 <= v < cam.intrinsics.height:
                    if hasattr(cam, 'mask') and cam.mask is not None:
                        mask_np = cam.mask.cpu().numpy() if isinstance(cam.mask, torch.Tensor) else cam.mask
                        if mask_np.ndim == 3: mask_np = mask_np[0]
                        mh, mw = mask_np.shape
                        su = int(u * (mw / cam.intrinsics.width))
                        sv = int(v * (mh / cam.intrinsics.height))
                        if 0 <= su < mw and 0 <= sv < mh and mask_np[sv, su] > 0.1:
                            hits += 1
                    else:
                        hits += 1

            if hits >= max(2, int(0.35 * len(cams))):
                valid_hull_pts.append(p3d)
                valid_hull_cols.append(np.array([0.85, 0.85, 0.85], dtype=np.float32))

        if len(valid_hull_pts) > 0:
            hull_pts = np.array(valid_hull_pts, dtype=np.float32)
            hull_cols = np.array(valid_hull_cols, dtype=np.float32)
            merged_pts = np.vstack([points, hull_pts])
            merged_cols = np.vstack([colors, hull_cols])
            print(f"[OK] Visual Hull Seeding: added {len(hull_pts):,} volumetric seed points (Total: {len(merged_pts):,})")
            return merged_pts, merged_cols

        return points, colors

    def _remove_statistical_outliers(
        self,
        points: np.ndarray,
        colors: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove statistical outliers based on mean KNN distance using fast cKDTree."""
        if len(points) < self.k_neighbors + 1:
            return points, colors

        tree = cKDTree(points)
        dists, _ = tree.query(points, k=self.k_neighbors + 1, workers=-1)
        mean_dists = dists[:, 1:].mean(axis=-1)

        mu = np.mean(mean_dists)
        std = np.std(mean_dists)
        threshold = mu + self.outlier_std_ratio * std

        valid_mask = mean_dists <= threshold
        return points[valid_mask], colors[valid_mask]

    def _compute_knn_distances(self, points: np.ndarray) -> np.ndarray:
        """Compute mean distance to k nearest neighbors for spatial scale initialization."""
        if len(points) < self.k_neighbors + 1:
            return np.full((len(points),), 0.05, dtype=np.float32)

        tree = cKDTree(points)
        dists, _ = tree.query(points, k=self.k_neighbors + 1, workers=-1)
        mean_dists = dists[:, 1:].mean(axis=-1)
        return np.maximum(mean_dists, 1e-6)

    def _estimate_normals(self, points: np.ndarray) -> np.ndarray:
        """Estimate surface normal vectors using local PCA covariance decomposition via cKDTree."""
        N = len(points)
        normals = np.zeros((N, 3), dtype=np.float32)

        if N < 4:
            normals[:, 1] = 1.0
            return normals

        k = min(12, N)
        tree = cKDTree(points)
        _, indices = tree.query(points, k=k, workers=-1)

        for i in range(N):
            neighbors = points[indices[i]]
            center = np.mean(neighbors, axis=0)
            cov = np.cov((neighbors - center).T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            normal = eigenvectors[:, 0]
            normal_norm = np.linalg.norm(normal)
            if normal_norm > 1e-6:
                normal /= normal_norm
            normals[i] = normal

        return normals
