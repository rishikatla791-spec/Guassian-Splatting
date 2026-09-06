"""
Scene: high-level container for a 3DGS dataset (cameras + point cloud).
"""
from __future__ import annotations
from typing import List, Optional

import numpy as np
import torch


class Scene:
    """
    Holds all cameras and the initial point cloud for a 3DGS scene.

    Attributes:
        train_cameras: list of Camera objects for training
        test_cameras:  list of Camera objects for evaluation
        point_cloud:   initial sparse points (PointCloud or None)
    """

    def __init__(
        self,
        train_cameras: List,
        test_cameras: List,
        points3d: Optional[np.ndarray] = None,
        colors: Optional[np.ndarray] = None,
        scene_info: Optional[dict] = None,
    ):
        self.train_cameras = train_cameras
        self.test_cameras = test_cameras
        self.points3d = points3d      # (N, 3)
        self.colors = colors          # (N, 3)
        self.scene_info = scene_info or {}

    def get_scene_extent(self) -> float:
        """
        Compute the spatial extent of the scene.

        Defined as the maximum distance from the scene centroid to any
        training camera center. Used to set scale thresholds in densification.

        Returns:
            Scalar extent (float)
        """
        if not self.train_cameras:
            return 1.0
        centers = np.stack([
            cam.extrinsics.camera_center for cam in self.train_cameras
        ])  # (N, 3)
        centroid = centers.mean(axis=0)
        dists = np.linalg.norm(centers - centroid, axis=-1)
        return float(dists.max())

    def get_center(self) -> np.ndarray:
        """Return median camera position as scene center."""
        if not self.train_cameras:
            return np.zeros(3)
        centers = np.stack([
            cam.extrinsics.camera_center for cam in self.train_cameras
        ])
        return np.median(centers, axis=0)

    def get_train_cameras(self, resolution_scale: float = 1.0) -> List:
        """Return training cameras (resolution_scale reserved for future downscaling)."""
        return self.train_cameras

    def get_test_cameras(self, resolution_scale: float = 1.0) -> List:
        """Return evaluation cameras."""
        return self.test_cameras

    def __len__(self) -> int:
        return len(self.train_cameras)

    def __repr__(self) -> str:
        n_pts = len(self.points3d) if self.points3d is not None else 0
        return (f"Scene("
                f"train={len(self.train_cameras)}, "
                f"test={len(self.test_cameras)}, "
                f"pts={n_pts}, "
                f"extent={self.get_scene_extent():.3f})")
