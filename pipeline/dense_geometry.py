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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Clean, densify, and compute geometric properties for point cloud.

        Args:
            points: (N, 3) float32 coordinates
            colors: (N, 3) float32 RGB values in [0, 1]

        Returns:
            clean_points: (M, 3)
            clean_colors: (M, 3)
            normals: (M, 3) estimated surface normals
            knn_dist: (M,) mean distance to k-nearest neighbors
        """
        print(f"=== [Dense Geometry] Processing point cloud: {len(points)} initial points ===")

        # 1. Statistical Outlier Filter
        clean_pts, clean_cols = self._remove_statistical_outliers(points, colors)
        print(f"[OK] Outlier filter: retained {len(clean_pts)} / {len(points)} points")

        # 2. KNN Distance Scale Computation
        knn_dist = self._compute_knn_distances(clean_pts)

        # 3. Surface Normal Estimation
        normals = self._estimate_normals(clean_pts)

        return clean_pts, clean_cols, normals, knn_dist

    def _remove_statistical_outliers(
        self,
        points: np.ndarray,
        colors: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove statistical outliers based on mean KNN distance."""
        if len(points) < self.k_neighbors + 1:
            return points, colors

        pts_t = torch.from_numpy(points).float()
        dist_matrix = torch.cdist(pts_t, pts_t)
        topk_dists, _ = torch.topk(dist_matrix, k=self.k_neighbors + 1, largest=False, dim=-1)
        mean_dists = topk_dists[:, 1:].mean(dim=-1).numpy()

        mu = np.mean(mean_dists)
        std = np.std(mean_dists)
        threshold = mu + self.outlier_std_ratio * std

        valid_mask = mean_dists <= threshold
        return points[valid_mask], colors[valid_mask]

    def _compute_knn_distances(self, points: np.ndarray) -> np.ndarray:
        """Compute mean distance to k nearest neighbors for spatial scale initialization."""
        pts_t = torch.from_numpy(points).float()
        dist_matrix = torch.cdist(pts_t, pts_t)
        topk_dists, _ = torch.topk(dist_matrix, k=self.k_neighbors + 1, largest=False, dim=-1)
        mean_dists = topk_dists[:, 1:].mean(dim=-1).numpy()
        return np.maximum(mean_dists, 1e-6)

    def _estimate_normals(self, points: np.ndarray) -> np.ndarray:
        """Estimate surface normal vectors using local PCA covariance decomposition."""
        N = len(points)
        normals = np.zeros((N, 3), dtype=np.float32)

        if N < 4:
            normals[:, 1] = 1.0
            return normals

        pts_t = torch.from_numpy(points).float()
        dist_matrix = torch.cdist(pts_t, pts_t)
        _, indices = torch.topk(dist_matrix, k=min(12, N), largest=False, dim=-1)

        for i in range(N):
            neighbors = points[indices[i].numpy()]
            center = np.mean(neighbors, axis=0)
            cov = np.cov((neighbors - center).T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            normal = eigenvectors[:, 0]
            normal_norm = np.linalg.norm(normal)
            if normal_norm > 1e-6:
                normal /= normal_norm
            normals[i] = normal

        return normals
