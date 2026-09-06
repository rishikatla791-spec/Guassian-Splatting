"""
Dynamic Hardware Auto-Detection & Memory Budget Manager.

Automatically queries GPU hardware, total VRAM, CUDA Compute Capability, and System RAM at runtime.
Adapts VRAM allocation limits, Gaussian capacity, tile chunk sizes, and execution precision dynamically
without hard-coded ceilings.
"""
from __future__ import annotations

import gc
import os
from typing import Dict, List, Optional, Tuple

import torch


class VRAMBudgetManager:
    """
    Hardware-Adaptive Memory & VRAM Acceleration Manager.
    Automatically scales workloads and VRAM budgets based on detected hardware.
    """

    def __init__(self, safe_vram_fraction: float = 0.85, enable_amp: bool = True):
        self.safe_vram_fraction = safe_vram_fraction
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_amp = enable_amp and torch.cuda.is_available()

        # Dynamic Hardware Auto-Detection
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(self.device)
            self.device_name = props.name
            self.total_vram_bytes = props.total_memory
            self.total_vram_gb = props.total_memory / (1024 ** 3)
            self.compute_capability = (props.major, props.minor)
            self.multi_processor_count = getattr(props, "multi_processor_count", 1)
        else:
            self.device_name = "CPU Only"
            self.total_vram_bytes = 0
            self.total_vram_gb = 0.0
            self.compute_capability = (0, 0)
            self.multi_processor_count = 1

        # Target safe VRAM limit (85% of actual detected GPU VRAM by default)
        self.max_vram_bytes = int(self.total_vram_bytes * self.safe_vram_fraction)
        self.max_vram_gb = self.max_vram_bytes / (1024 ** 3)

    def get_vram_stats(self) -> Dict[str, float]:
        """
        Get current GPU VRAM utilization statistics.

        Returns:
            dict with 'allocated_gb', 'reserved_gb', 'max_allocated_gb', 'total_vram_gb', 'safe_limit_gb'
        """
        if not torch.cuda.is_available():
            return {
                "allocated_gb": 0.0,
                "reserved_gb": 0.0,
                "max_allocated_gb": 0.0,
                "total_vram_gb": 0.0,
                "safe_limit_gb": 0.0,
            }

        alloc = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
        res = torch.cuda.memory_reserved(self.device) / (1024 ** 3)
        max_alloc = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)

        return {
            "allocated_gb": alloc,
            "reserved_gb": res,
            "max_allocated_gb": max_alloc,
            "total_vram_gb": self.total_vram_gb,
            "safe_limit_gb": self.max_vram_gb,
        }

    def compute_max_gaussian_capacity(self, bytes_per_gaussian: int = 944) -> int:
        """
        Dynamically compute maximum safe Gaussian capacity for the detected GPU VRAM.

        Args:
            bytes_per_gaussian: Memory required per Gaussian during training
                                (State ~236B + Adam Optimizer States ~708B = ~944B)
        """
        if not torch.cuda.is_available():
            return 100_000 # Safe CPU default

        # Allocate 60% of safe VRAM budget for Gaussian parameters and optimizer states,
        # leaving 40% for tile sorting workspace, image framebuffers, and autograd activations.
        available_state_vram = self.max_vram_bytes * 0.60
        max_capacity = int(available_state_vram / bytes_per_gaussian)
        return max(max_capacity, 50_000)

    def print_memory_status(self, label: str = "") -> None:
        """Print formatted GPU VRAM memory status based on detected device properties."""
        stats = self.get_vram_stats()
        if torch.cuda.is_available():
            print(f"  [VRAM Monitor {label}] Device: {self.device_name} | "
                  f"Allocated: {stats['allocated_gb']:.2f} GB | "
                  f"Peak: {stats['max_allocated_gb']:.2f} GB / {self.total_vram_gb:.2f} GB (Safe Ceiling: {self.max_vram_gb:.2f} GB)")
        else:
            print(f"  [VRAM Monitor {label}] Running on CPU (No GPU CUDA device detected)")

    def optimize_memory(self, force: bool = False) -> None:
        """
        Perform PyTorch garbage collection and CUDA cache release if VRAM exceeds safe limit.
        """
        if torch.cuda.is_available():
            stats = self.get_vram_stats()
            if force or stats["allocated_gb"] > (self.max_vram_gb * 0.90):
                gc.collect()
                torch.cuda.empty_cache()

    def get_amp_autocast(self):
        """
        Return PyTorch Automatic Mixed Precision (AMP FP16) context manager.
        """
        if self.enable_amp and torch.cuda.is_available():
            return torch.cuda.amp.autocast(dtype=torch.float16)
        else:
            class DummyContext:
                def __enter__(self): pass
                def __exit__(self, exc_type, exc_val, exc_tb): pass
            return DummyContext()

    def compute_optimal_chunk_size(self, num_gaussians: int) -> int:
        """
        Compute optimal chunk size for tile rasterization based on runtime VRAM capacity.
        """
        if self.total_vram_gb >= 12.0:
            return num_gaussians  # Process full scene without chunking on high-end GPUs
        elif self.total_vram_gb >= 6.0:
            return min(num_gaussians, 200_000)
        else:
            return min(num_gaussians, 50_000)
