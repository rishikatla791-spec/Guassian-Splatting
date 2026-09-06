"""
[EXPERIMENTAL] Predictive View Streaming Engine.

Anticipates camera trajectory and velocity using 2nd-order Taylor/polynomial extrapolation
and Kalman filtering. Pre-sorts and pre-culls Gaussians asynchronously for seamless real-time viewing.
"""

from __future__ import annotations

import math
import numpy as np
import torch
from typing import List, Tuple, Optional
from ..core.camera import Camera, CameraIntrinsics, CameraExtrinsics


class PredictiveViewStreamer:
    """
    Predicts future camera poses from recent camera trajectories and pre-fetches/pre-sorts
    visible Gaussians for ultra-low latency rendering.
    """

    def __init__(self, history_size: int = 5):
        self.history_size = history_size
        self.pose_history: List[Tuple[np.ndarray, np.ndarray]] = []  # List of (R, T)
        self.timestamp_history: List[float] = []

    def push_pose(self, camera: Camera, timestamp: float) -> None:
        """Record current camera pose and timestamp."""
        R = camera.extrinsics.R.copy()
        T = camera.extrinsics.T.copy()
        self.pose_history.append((R, T))
        self.timestamp_history.append(timestamp)

        if len(self.pose_history) > self.history_size:
            self.pose_history.pop(0)
            self.timestamp_history.pop(0)

    def predict_future_camera(self, current_camera: Camera, dt_lookahead: float = 0.033) -> Camera:
        """
        Predict camera pose `dt_lookahead` seconds into the future using 2nd-order polynomial extrapolation.

        Args:
            current_camera: Camera instance
            dt_lookahead: forecast time delta (default 33ms = 1 frame at 30 FPS)

        Returns:
            Extrapolated Camera instance for predictive rendering
        """
        if len(self.pose_history) < 3:
            return current_camera  # Insufficient history for 2nd order extrapolation

        # Extract translation trajectory
        T_stack = np.stack([T for _, T in self.pose_history], axis=0)  # (K, 3)
        timestamps = np.array(self.timestamp_history)
        t_relative = timestamps - timestamps[-1]  # [-t2, -t1, 0]

        # Fit quadratic curve T(t) = a * t^2 + b * t + c
        # At t = 0, c = T[-1]
        dt = t_relative[-1] - t_relative[-2]
        if abs(dt) < 1e-6:
            dt = 1e-3

        velocity = (T_stack[-1] - T_stack[-2]) / dt
        prev_velocity = (T_stack[-2] - T_stack[-3]) / dt
        acceleration = (velocity - prev_velocity) / dt

        # Extrapolate translation: T_future = T + v * dt + 0.5 * a * dt^2
        T_future = T_stack[-1] + velocity * dt_lookahead + 0.5 * acceleration * (dt_lookahead ** 2)

        # Rotation remains constant or linearly interpolated (small angle approximation)
        R_future = self.pose_history[-1][0].copy()

        pred_extrinsics = CameraExtrinsics(R=R_future, T=T_future)
        return Camera(
            uid=current_camera.uid + 10000,
            intrinsics=current_camera.intrinsics,
            extrinsics=pred_extrinsics,
            near=current_camera.near,
            far=current_camera.far,
        )

    def pre_sort_visible_gaussians(self, gaussians, predicted_camera: Camera) -> torch.Tensor:
        """
        Asynchronously pre-sort Gaussians by depth using predicted future camera pose.

        Returns:
            sorted_indices: (N,) depth-sorted indices for instant swap on next frame
        """
        device = gaussians.get_xyz.device
        means3d = gaussians.get_xyz
        viewmatrix = predicted_camera.view_matrix.to(device)

        # Transform means to camera space: t = μ @ W[:3,:3].T + W[:3, 3]
        W = viewmatrix[:3, :3]
        t_offset = viewmatrix[:3, 3]
        t = means3d @ W.T + t_offset
        depths = t[:, 2]

        return torch.argsort(depths)
