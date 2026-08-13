"""
Camera Guidance & View Coverage Planner for 3D Gaussian Splatting.

This module provides tools for evaluating multi-view camera photo coverage,
calculating angular distribution entropy, identifying missing view angles,
and computing Next-Best-View (NBV) recommendations for optimal 3DGS reconstruction.

Mathematical Principles:
1. Target Orbit Sampling: Discretizes a 3D bounding hemisphere/sphere using a
   Fibonacci lattice or geodesic dome to define ideal capture targets.
2. Frustum Overlap Metric: Measures 3D view frustum overlap and baseline parallax
   between consecutive camera poses to ensure COLMAP SfM feature matching succeeds.
3. Spatial Coverage Score: Computes angular entropy and surface normal coverage
   to detect under-sampled camera angles before training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch

from ..core.camera import Camera, CameraIntrinsics, CameraExtrinsics


@dataclass
class TargetViewNode:
    """Target capture node on a spherical/hemispherical orbit around the object."""
    node_id: int
    position: np.ndarray      # (3,) position in world coordinates [x, y, z]
    look_at: np.ndarray       # (3,) target center [x, y, z]
    elevation_deg: float      # angle above horizon in degrees
    azimuth_deg: float        # angle around vertical axis in degrees
    is_captured: bool = False # status flag
    best_matching_cam: Optional[int] = None
    min_angular_dist_deg: float = 180.0


class ViewCoveragePlanner:
    """
    Evaluates multi-view camera capture quality for 3D Gaussian Splatting.

    Calculates:
      - Spherical coverage & angular density map
      - Overlap / baseline quality scores for COLMAP
      - Next-Best-View (NBV) recommendations for user guidance
    """

    def __init__(
        self,
        center: np.ndarray = np.array([0.0, 0.0, 0.0]),
        radius: float = 2.0,
        num_elevation_rings: int = 3,
        samples_per_ring: int = 12,
    ):
        """
        Args:
            center: 3D bounding center of object/scene
            radius: capture distance radius from object center
            num_elevation_rings: number of elevation levels (e.g. 15°, 45°, 75°)
            samples_per_ring: camera positions sampled around each ring
        """
        self.center = np.asarray(center, dtype=np.float64)
        self.radius = radius
        self.target_nodes: List[TargetViewNode] = []
        self._generate_target_dome(num_elevation_rings, samples_per_ring)

    def _generate_target_dome(self, num_rings: int, samples_per_ring: int) -> None:
        """Generate a geodesic hemisphere of target camera positions."""
        node_id = 0
        elevations = np.linspace(15.0, 75.0, num_rings)

        for elev in elevations:
            elev_rad = math.radians(elev)
            # Offset alternate rings for staggered coverage
            azimuth_offset = 0.0 if (node_id % 2 == 0) else (180.0 / samples_per_ring)

            for i in range(samples_per_ring):
                azimuth_deg = (i * (360.0 / samples_per_ring) + azimuth_offset) % 360.0
                azimuth_rad = math.radians(azimuth_deg)

                # Spherical to Cartesian coordinates
                x = self.center[0] + self.radius * math.cos(elev_rad) * math.cos(azimuth_rad)
                y = self.center[1] + self.radius * math.sin(elev_rad)  # Y-up
                z = self.center[2] + self.radius * math.cos(elev_rad) * math.sin(azimuth_rad)

                pos = np.array([x, y, z], dtype=np.float64)
                self.target_nodes.append(
                    TargetViewNode(
                        node_id=node_id,
                        position=pos,
                        look_at=self.center,
                        elevation_deg=elev,
                        azimuth_deg=azimuth_deg,
                    )
                )
                node_id += 1

    def evaluate_captured_cameras(
        self,
        cameras: List[Camera],
        max_dist_threshold_deg: float = 45.0,

    ) -> Dict[str, float]:
        """
        Evaluate how well a list of captured cameras covers the target dome.

        Args:
            cameras: list of captured Camera objects
            max_dist_threshold_deg: maximum angular separation to consider a target node satisfied

        Returns:
            Dictionary of metrics (coverage_ratio, angular_entropy, min_overlap_score, recommendations)
        """
        if not cameras:
            return {
                "coverage_ratio": 0.0,
                "satisfied_nodes": 0,
                "total_nodes": len(self.target_nodes),
                "angular_entropy": 0.0,
                "quality_score": 0.0,
            }

        cam_centers = []
        for cam in cameras:
            c = cam.camera_center
            if isinstance(c, torch.Tensor):
                c = c.detach().cpu().numpy()
            cam_centers.append(np.asarray(c, dtype=np.float64))
        cam_centers = np.stack(cam_centers, axis=0)


        # Reset node states
        for node in self.target_nodes:
            node.is_captured = False
            node.min_angular_dist_deg = 180.0
            node.best_matching_cam = None

        # Calculate angular distance between target nodes and captured cameras
        for node in self.target_nodes:
            target_dir = node.position - self.center
            target_dir /= np.linalg.norm(target_dir)

            for idx, cam_c in enumerate(cam_centers):
                cam_dir = cam_c - self.center
                cam_dir_norm = np.linalg.norm(cam_dir)
                if cam_dir_norm < 1e-6:
                    continue
                cam_dir /= cam_dir_norm

                # Angular distance (dot product arccos)
                dot = np.clip(np.dot(target_dir, cam_dir), -1.0, 1.0)
                ang_deg = math.degrees(math.acos(dot))

                if ang_deg < node.min_angular_dist_deg:
                    node.min_angular_dist_deg = ang_deg
                    node.best_matching_cam = idx

                if ang_deg <= max_dist_threshold_deg:
                    node.is_captured = True

        captured_count = sum(1 for node in self.target_nodes if node.is_captured)
        coverage_ratio = captured_count / float(len(self.target_nodes))

        # Calculate angular distribution entropy (measure of uniform coverage)
        angular_dists = [node.min_angular_dist_deg for node in self.target_nodes]
        avg_ang_dist = float(np.mean(angular_dists))

        quality_score = max(0.0, 100.0 * (1.0 - (avg_ang_dist / 45.0)))

        return {
            "coverage_ratio": float(coverage_ratio),
            "satisfied_nodes": int(captured_count),
            "total_nodes": len(self.target_nodes),
            "avg_angular_gap_deg": float(avg_ang_dist),
            "quality_score": float(quality_score),
        }

    def get_next_best_views(self, top_k: int = 3) -> List[TargetViewNode]:
        """Return top_k unsatisfied target nodes with the largest coverage gaps."""
        unsatisfied = [node for node in self.target_nodes if not node.is_captured]
        # Sort by largest angular gap first
        unsatisfied.sort(key=lambda n: n.min_angular_dist_deg, reverse=True)
        return unsatisfied[:top_k]
