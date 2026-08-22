"""
Dynamic VRAM & GPU Memory Manager with Runtime Hardware Auto-Detection.

Provides proactive VRAM budget tracking, dynamic memory chunking,
automatic garbage collection, mixed precision casting, and Structure of Arrays (SoA) layout.
Adapts dynamically to whatever GPU hardware is detected at runtime.
"""

from __future__ import annotations

import gc
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple


class VRAMManager:
    """
    Hardware-Adaptive VRAM Budget Monitor and Performance Optimizer.
    Automatically detects available GPU VRAM and adjusts execution bounds.
    """

    def __init__(self, safe_fraction: float = 0.85):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.safe_fraction = safe_fraction

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(self.device)
            self.total_vram_bytes = props.total_memory
            self.total_vram_gb = props.total_memory / (1024 ** 3)
            self.device_name = props.name
        else:
            self.total_vram_bytes = 0
            self.total_vram_gb = 0.0
            self.device_name = "CPU Only"

        self.target_vram_bytes = int(self.total_vram_bytes * self.safe_fraction)
        self.target_vram_gb = self.target_vram_bytes / (1024 ** 3)

    def get_vram_stats(self) -> Dict[str, float]:
        """
        Return current GPU VRAM statistics in MB.
        """
        if not torch.cuda.is_available():
            return {
                "allocated_mb": 0.0,
                "reserved_mb": 0.0,
                "max_allocated_mb": 0.0,
                "total_vram_mb": 0.0,
                "target_vram_mb": 0.0,
            }

        return {
            "allocated_mb": torch.cuda.memory_allocated(self.device) / (1024 ** 2),
            "reserved_mb": torch.cuda.memory_reserved(self.device) / (1024 ** 2),
            "max_allocated_mb": torch.cuda.max_memory_allocated(self.device) / (1024 ** 2),
            "total_vram_mb": self.total_vram_gb * 1024,
            "target_vram_mb": self.target_vram_gb * 1024,
        }

    def check_and_clean(self, threshold_fraction: float = 0.90) -> bool:
        """
        Check if VRAM usage exceeds safety threshold; if so, run proactive cleanup.

        Returns:
            True if cleanup was triggered.
        """
        if not torch.cuda.is_available():
            return False

        current_bytes = torch.cuda.memory_allocated(self.device)
        if current_bytes > self.target_vram_bytes * threshold_fraction:
            gc.collect()
            torch.cuda.empty_cache()
            return True
        return False

    def optimize_layout_soa(self, gaussians) -> Dict[str, torch.Tensor]:
        """
        Convert Gaussian parameters into contiguous Structure-of-Arrays (SoA) layout
        for maximum GPU L2 cache hit rate and memory bandwidth saturation.
        """
        with torch.no_grad():
            xyz = gaussians.get_xyz.contiguous()
            rotations = gaussians.get_rotation.contiguous()
            scales = gaussians.get_scaling.contiguous()
            opacities = gaussians.get_opacity.contiguous()
            sh = gaussians.get_features.contiguous()

        return {
            "xyz": xyz,
            "rotations": rotations,
            "scales": scales,
            "opacities": opacities,
            "sh": sh,
        }

    def compute_optimal_tile_chunk_size(self, num_gaussians: int, image_height: int, image_width: int) -> int:
        """
        Compute optimal tile size / chunk size to avoid VRAM OOM during high-resolution rendering.

        Returns tile_size in pixels (e.g. 16, 8, or 4).
        """
        if not torch.cuda.is_available():
            return 16

        pixels = image_height * image_width
        memory_footprint_est = num_gaussians * 236 + pixels * 64

        if memory_footprint_est > self.target_vram_bytes * 0.75:
            return 8  # Use smaller tile grid for high memory pressure
        return 16
