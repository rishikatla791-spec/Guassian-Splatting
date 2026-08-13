"""
[EXPERIMENTAL] Adaptive Quality & FPS Controller.

Closed-loop dynamic control loop targeting specified target FPS (e.g. 60 FPS on RTX 3050).
Dynamically tunes LOD level, max active SH degree, tile size, and alpha threshold based on frame rendering latency.
"""

from __future__ import annotations

import time
import torch
from typing import Dict, Any, Tuple


class AdaptiveFPSController:
    """
    Closed-loop feedback controller for maintaining constant target framerate.

    Controls:
      - active_sh_degree: [0, 1, 2, 3] (lower = faster evaluation)
      - radius_threshold: [2.0, 3.0, 4.0] (lower = tighter bounding boxes)
      - alpha_threshold:  [1/255, 5/255, 10/255] (higher = earlier termination)
      - lod_level_bias:   [0, 1, 2] (higher = coarser LOD)
    """

    def __init__(self, target_fps: float = 60.0, alpha_gain: float = 0.1):
        self.target_fps = target_fps
        self.target_frame_time = 1.0 / target_fps  # e.g., 0.01667 s for 60 FPS
        self.alpha_gain = alpha_gain

        self.active_sh_degree = 3
        self.radius_threshold = 3.0
        self.alpha_threshold = 1.0 / 255.0
        self.lod_bias = 0

        self.latency_history = []

    def update(self, frame_latency_s: float) -> Dict[str, Any]:
        """
        Update controller with measured rendering latency of the last frame.

        Args:
            frame_latency_s: measured render time in seconds

        Returns:
            Dict containing current adaptive parameter state
        """
        self.latency_history.append(frame_latency_s)
        if len(self.latency_history) > 10:
            self.latency_history.pop(0)

        avg_latency = sum(self.latency_history) / len(self.latency_history)

        # PID-like adjustment
        error = avg_latency - self.target_frame_time

        if error > 0.005:  # Frame is too slow (latency > target + 5ms)
            # Downgrade settings to speed up rendering
            if self.active_sh_degree > 0 and error > 0.015:
                self.active_sh_degree -= 1
            elif self.alpha_threshold < 0.05:
                self.alpha_threshold += 0.005
            elif self.radius_threshold > 2.0:
                self.radius_threshold -= 0.2
            elif self.lod_bias < 3:
                self.lod_bias += 1

        elif error < -0.005:  # Frame is fast (headroom available)
            # Upgrade settings for higher visual quality
            if self.lod_bias > 0:
                self.lod_bias -= 1
            elif self.radius_threshold < 3.0:
                self.radius_threshold += 0.2
            elif self.alpha_threshold > 1.0 / 255.0:
                self.alpha_threshold = max(1.0 / 255.0, self.alpha_threshold - 0.002)
            elif self.active_sh_degree < 3 and error < -0.010:
                self.active_sh_degree += 1

        return {
            "current_fps": 1.0 / max(avg_latency, 1e-6),
            "sh_degree": self.active_sh_degree,
            "radius_threshold": self.radius_threshold,
            "alpha_threshold": self.alpha_threshold,
            "lod_bias": self.lod_bias,
        }
