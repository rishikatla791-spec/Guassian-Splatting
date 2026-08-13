"""
VRAM & GPU Memory Manager for RTX 3050 6GB + 16GB System RAM.

Provides proactive VRAM budget tracking, dynamic memory chunking,
automatic garbage collection, mixed precision casting, and Structure of Arrays (SoA) layout.
"""

from __future__ import annotations

import gc
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple


class VRAMManager:
    """
    Proactive VRAM budget monitor and memory optimizer for GPU execution.

    Target GPU: RTX 3050 6GB VRAM (Safety limit: ~5.0 GB peak allocation).
    """

    def __init__(self, target_vram_gb: float = 5.0):
        self.target_vram_bytes = int(target_vram_gb * (1024 ** 3))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_vram_stats(self) -> Dict[str, float]:
        """
        Return current GPU VRAM statistics in MB.
        """
        if not torch.cuda.is_available():
            return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}

        return {
            "allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
            "reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2),
            "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
        }

    def check_and_clean(self, threshold_fraction: float = 0.85) -> bool:
        """
        Check if VRAM usage exceeds safety threshold; if so, run proactive cleanup.

        Returns:
            True if cleanup was triggered.
        """
        if not torch.cuda.is_available():
            return False

        current_bytes = torch.cuda.memory_allocated()
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
        pixels = image_height * image_width
        memory_footprint_est = num_gaussians * 236 + pixels * 64

        if memory_footprint_est > self.target_vram_bytes * 0.7:
            return 8  # Use smaller tiles for high memory pressure
        return 16
