"""
[EXPERIMENTAL] Self-Optimizing Gaussian Allocator.

Maintains a fixed budget B of Gaussians and dynamically reallocates capacity
toward the most important scene regions.

Importance scoring:
  importance_i = opacity_i · grad_norm_i · (1 / scale_mean_i)

High importance → Gaussian captures a complex, high-frequency region.
Low importance  → Gaussian is redundant or in a flat region.

Memory estimation:
  Per-Gaussian storage:
    xyz:       3 × 4 = 12 bytes
    quaternion:4 × 4 = 16 bytes
    scale:     3 × 4 = 12 bytes
    opacity:   1 × 4 =  4 bytes
    SH coeff: 48 × 4 = 192 bytes  (degree 3: 16 × 3 channels)
    Total:                236 bytes/Gaussian  ≈ 0.231 KB/Gaussian
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
from typing import Optional


# Bytes per Gaussian (float32 storage, SH degree 3)
_BYTES_PER_GAUSSIAN = (3 + 4 + 3 + 1 + 48) * 4   # = 236 bytes


class SelfOptimizingAllocator:
    """
    Dynamically prunes and reallocates Gaussians within a fixed budget.

    Usage:
        allocator = SelfOptimizingAllocator(budget=500_000)
        allocator.update_importance(gaussians, gradient_accum)
        allocator.reallocate(gaussians, optimizer, target_count=400_000)
    """

    def __init__(self, budget: int = 1_000_000):
        """
        Args:
            budget: maximum number of Gaussians to maintain
        """
        self.budget = budget
        self._importance: Optional[torch.Tensor] = None

    # -----------------------------------------------------------------------
    # Importance scoring
    # -----------------------------------------------------------------------

    def update_importance(
        self,
        gaussians,
        gradient_accum: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute and cache per-Gaussian importance scores.

        importance_i = sigmoid(opacity_raw_i) · grad_norm_i · (1 / (scale_mean_i + ε))

        Args:
            gaussians:       GaussianModel instance
            gradient_accum:  (N, 1) accumulated |∇_xy| sums (from densification stats)

        Returns:
            (N,) importance scores (not normalized)
        """
        N = gaussians.num_gaussians
        device = gaussians.get_xyz.device

        opacity = gaussians.get_opacity.squeeze(-1).detach()  # (N,) in [0,1]
        scale_mean = gaussians.get_scaling.mean(dim=-1).detach().clamp(min=1e-8)  # (N,)

        if gradient_accum is not None:
            grad_norm = gradient_accum.squeeze(-1).detach().float()
            # Normalize to [0, 1]
            g_max = grad_norm.max().clamp(min=1e-8)
            grad_norm = grad_norm / g_max
        else:
            grad_norm = torch.ones(N, device=device)

        importance = opacity * grad_norm * (1.0 / scale_mean)
        self._importance = importance
        return importance

    # -----------------------------------------------------------------------
    # Reallocation
    # -----------------------------------------------------------------------

    def reallocate(
        self,
        gaussians,
        optimizer,
        target_count: Optional[int] = None,
    ) -> None:
        """
        Prune the lowest-importance Gaussians to reach target_count.

        Strategy:
          1. Prune bottom (N - target_count) by importance score
          2. The pruned budget can be filled by external densification

        Args:
            gaussians:    GaussianModel instance
            optimizer:    Adam optimizer (states updated in-place)
            target_count: target Gaussian count. Defaults to self.budget.
        """
        N = gaussians.num_gaussians
        target = target_count if target_count is not None else self.budget

        if N <= target:
            return  # Nothing to prune

        if self._importance is None:
            self.update_importance(gaussians)

        importance = self._importance
        n_prune = N - target

        # Find indices of n_prune least important Gaussians
        _, prune_idx = torch.topk(importance, k=n_prune, largest=False)
        prune_mask = torch.zeros(N, dtype=torch.bool, device=importance.device)
        prune_mask[prune_idx] = True

        gaussians._prune_gaussians(prune_mask, optimizer)
        gaussians._init_densification_stats()
        self._importance = None  # Reset cache

        print(f"[SelfOptimizingAllocator] Pruned {n_prune} Gaussians "
              f"({N} → {gaussians.num_gaussians})")

    # -----------------------------------------------------------------------
    # Memory utilities
    # -----------------------------------------------------------------------

    @staticmethod
    def estimate_memory(n_gaussians: int, sh_degree: int = 3) -> int:
        """
        Estimate GPU memory in bytes for a given number of Gaussians.

        Layout (float32):
          xyz:       3
          rotation:  4  (quaternion)
          scale:     3
          opacity:   1
          SH coeff: (sh_degree+1)² × 3

        Args:
            n_gaussians: number of Gaussians
            sh_degree:   SH degree (0..3)

        Returns:
            bytes (int)
        """
        n_sh = (sh_degree + 1) ** 2 * 3
        floats_per_gaussian = 3 + 4 + 3 + 1 + n_sh
        return n_gaussians * floats_per_gaussian * 4   # float32 = 4 bytes

    @staticmethod
    def max_gaussians_for_memory(
        vram_gb: float,
        sh_degree: int = 3,
        utilization: float = 0.8,
    ) -> int:
        """
        Estimate maximum Gaussian count for a given VRAM budget.

        Args:
            vram_gb:     total VRAM in gigabytes
            sh_degree:   SH degree (affects storage per Gaussian)
            utilization: fraction of VRAM to use (default 0.8)

        Returns:
            maximum number of Gaussians
        """
        available_bytes = int(vram_gb * 1024**3 * utilization)
        n_sh = (sh_degree + 1) ** 2 * 3
        bytes_per = (3 + 4 + 3 + 1 + n_sh) * 4
        return available_bytes // bytes_per

    def memory_report(self, gaussians) -> str:
        """Return a formatted memory usage report."""
        N = gaussians.num_gaussians
        sh = gaussians.sh_degree
        used = self.estimate_memory(N, sh)
        budget_mem = self.estimate_memory(self.budget, sh)
        return (f"Gaussians: {N:,} / {self.budget:,} budget\n"
                f"Memory:    {used/1024**2:.1f} MB used / "
                f"{budget_mem/1024**2:.1f} MB budget")
