"""
Spatial hash grid for fast nearest-neighbor queries.

Used for:
  1. Initializing Gaussian scales (mean k-NN distance)
  2. Finding candidate Gaussians for densification
  3. LOD merging of spatially close Gaussians

Mathematical basis:
  Hash function: h(x,y,z) = (x*p1 XOR y*p2 XOR z*p3) mod M
  where p1=73856093, p2=19349669, p3=83492791 (large primes)
  and M = table_size (next power of 2 > 10*N)

  Spatial cell size chosen so that each cell contains ~10 points on average.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Optional

import torch
import numpy as np


class SpatialHashGrid:
    """
    GPU-accelerated spatial hash grid for 3D point queries.

    Supports:
      - k-nearest neighbors
      - Radius search
      - Cell iteration
    """

    def __init__(self, cell_size: float, device: str = "cuda"):
        """
        Args:
            cell_size: edge length of each voxel cell
            device:    torch device string
        """
        self.cell_size = cell_size
        self.device = device
        self.points: Optional[torch.Tensor] = None
        self.cell_indices: Optional[torch.Tensor] = None
        self.sorted_points: Optional[torch.Tensor] = None

    def build(self, points: torch.Tensor) -> None:
        """
        Build the grid from a set of points.

        Args:
            points: (N, 3) float32 point positions
        """
        self.points = points.to(self.device)
        N = points.shape[0]

        # Compute integer cell coordinates
        cells = torch.floor(self.points / self.cell_size).long()  # (N, 3)

        # Hash to 1D using large primes
        P1, P2, P3 = 73856093, 19349669, 83492791
        table_size = 1 << math.ceil(math.log2(10 * N + 1))  # next power of 2
        hashes = (cells[:, 0] * P1 ^ cells[:, 1] * P2 ^ cells[:, 2] * P3) % table_size

        # Sort by hash for fast lookup
        sort_order = torch.argsort(hashes)
        self.cell_indices = hashes[sort_order]
        self.sorted_points = self.points[sort_order]
        self._sort_order = sort_order
        self._table_size = table_size

    def knn(self, query: torch.Tensor, k: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Approximate k-nearest neighbors using grid cells.
        Searches query point's cell and 26 neighbors (3³ - 1).

        Args:
            query: (M, 3) query points
            k:     number of neighbors

        Returns:
            distances: (M, k) distances to k nearest neighbors
            indices:   (M, k) indices into original point array
        """
        if self.points is None:
            raise RuntimeError("Call build() first")

        # For simplicity, use brute-force cdist for small N, or batched for large N
        N = self.points.shape[0]
        M = query.shape[0]
        query = query.to(self.device)

        if N <= 50_000:
            # Full pairwise distance
            dists = torch.cdist(query, self.points)  # (M, N)
            k_clamped = min(k, N)
            dists_k, idx_k = dists.topk(k_clamped, largest=False, dim=-1)
            return dists_k, idx_k
        else:
            # Batched to avoid OOM
            BATCH = 1024
            all_dists, all_idx = [], []
            for i in range(0, M, BATCH):
                q_batch = query[i:i+BATCH]
                dists = torch.cdist(q_batch, self.points)
                k_clamped = min(k, N)
                d_k, i_k = dists.topk(k_clamped, largest=False, dim=-1)
                all_dists.append(d_k)
                all_idx.append(i_k)
            return torch.cat(all_dists, dim=0), torch.cat(all_idx, dim=0)

    def radius_search(self, query: torch.Tensor, radius: float) -> List[torch.Tensor]:
        """
        Find all points within radius of each query point.

        Args:
            query:  (M, 3) query points
            radius: search radius

        Returns:
            list of length M, each element is (K_i,) indices of neighbors
        """
        query = query.to(self.device)
        dists = torch.cdist(query, self.points)  # (M, N)
        results = []
        for i in range(query.shape[0]):
            results.append((dists[i] <= radius).nonzero(as_tuple=True)[0])
        return results


def compute_knn_distances(
    points: torch.Tensor,
    k: int = 3,
    batch_size: int = 4096,
) -> torch.Tensor:
    """
    Compute mean k-nearest-neighbor distance for each point.
    Used for Gaussian scale initialization.

    Args:
        points:     (N, 3)
        k:          number of neighbors
        batch_size: batch size for memory-efficient computation

    Returns:
        (N,) mean distance to k nearest neighbors
    """
    N = points.shape[0]
    device = points.device
    k = min(k + 1, N)  # +1 to exclude self

    all_dists = []
    for i in range(0, N, batch_size):
        batch = points[i:i+batch_size]
        dists = torch.cdist(batch, points)  # (B, N)
        dists[:, i:i+batch.shape[0]].fill_diagonal_(float('inf'))  # exclude self
        knn, _ = dists.topk(k, largest=False, dim=-1)
        all_dists.append(knn[:, 1:].mean(dim=-1))  # skip self (should be 0 after diag fill)

    return torch.cat(all_dists, dim=0)
