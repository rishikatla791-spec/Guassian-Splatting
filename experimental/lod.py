"""
HierarchicalLOD: Octree-style Level-of-Detail for 3D Gaussian Splatting.

Gaussians are spatially clustered into hierarchical levels using k-means.
At render time, the appropriate detail level is selected per camera distance,
and Gaussians within a level are merged using the mean-of-Gaussian-mixtures
approximation:

  μ_merged  = Σ wᵢ μᵢ                     (volume-weighted mean)
  Σ_merged  = Σ wᵢ (Σᵢ + (μᵢ - μ̄)(μᵢ - μ̄)ᵀ)  (mixture covariance)

where wᵢ = Vᵢ / Σ Vᵢ  and  Vᵢ = det(Σᵢ)^(1/2) (Gaussian volume proxy).

LOD threshold: level k is selected when camera_dist > base_extent * 2^k.
Level 0 is full detail; increasing level = coarser.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.gaussians import GaussianModel
from ..core.camera import Camera
from ..core.math_utils import quaternion_to_rotation_matrix, rotation_matrix_to_quaternion


# ---------------------------------------------------------------------------
# Spatial k-means (pure torch, no sklearn dependency)
# ---------------------------------------------------------------------------

def _torch_kmeans(
    points: torch.Tensor,
    k: int,
    n_iter: int = 50,
    tol: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Lloyd's k-means algorithm implemented in pure PyTorch.

    Args:
        points: (N, D) float32 tensor of data points
        k:      number of clusters
        n_iter: maximum iterations
        tol:    centroid shift tolerance for early stopping

    Returns:
        centroids:   (k, D) cluster centroids
        assignments: (N,) long tensor of cluster indices in [0, k)
    """
    N, D = points.shape
    device = points.device

    # Initialise centroids via k-means++ seeding for numerical stability
    perm = torch.randperm(N, device=device)
    centroids = points[perm[:k]].clone()  # (k, D)

    for _ in range(n_iter):
        # Distances: (N, k)
        dists = torch.cdist(points, centroids)
        assignments = dists.argmin(dim=1)  # (N,)

        # Recompute centroids
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(k, device=device, dtype=torch.float32)
        new_centroids.scatter_add_(0, assignments.unsqueeze(1).expand(-1, D), points)
        counts.scatter_add_(0, assignments, torch.ones(N, device=device))

        # Guard empty clusters → keep old centroid
        mask = counts > 0
        new_centroids[mask] = new_centroids[mask] / counts[mask].unsqueeze(1)
        new_centroids[~mask] = centroids[~mask]

        shift = (new_centroids - centroids).norm()
        centroids = new_centroids
        if shift < tol:
            break

    return centroids, assignments


# ---------------------------------------------------------------------------
# Merged Gaussian helper
# ---------------------------------------------------------------------------

def _merge_cluster(
    gaussians: GaussianModel,
    indices: torch.Tensor,
) -> dict:
    """
    Merge a set of Gaussians (indexed by `indices`) into a single Gaussian
    using the mean-of-Gaussian-mixtures (MoG) approximation.

    Weights wᵢ are proportional to opacity × volume (det(Σ)^½).

    Returns a dict with keys matching GaussianModel raw parameter names:
        _xyz, _features_dc, _features_rest, _scaling, _rotation, _opacity
    """
    device = gaussians._xyz.device

    # Activated quantities
    mu    = gaussians.get_xyz[indices]          # (M, 3)
    alpha = gaussians.get_opacity[indices]      # (M, 1)
    scale = gaussians.get_scaling[indices]      # (M, 3)
    rot   = gaussians.get_rotation[indices]     # (M, 4)

    # Volume proxy: product of scales ≈ det(S) (since Σ = R S Sᵀ Rᵀ, det = s₀s₁s₂)
    vol = scale.prod(dim=1, keepdim=True)       # (M, 1)

    # Weights: opacity * volume, normalised
    w = (alpha * vol).squeeze(1)                # (M,)
    w = w / w.sum().clamp(min=1e-10)            # normalise

    # ---- Merged mean ---------------------------------------------------------
    mu_merged = (w.unsqueeze(1) * mu).sum(dim=0)  # (3,)

    # ---- Merged covariance (MoG formula) ------------------------------------
    # Σ_merged = Σ wᵢ (Σᵢ + δᵢ δᵢᵀ)   where δᵢ = μᵢ - μ_merged
    R = quaternion_to_rotation_matrix(rot)      # (M, 3, 3)
    S = torch.diag_embed(scale)                 # (M, 3, 3) scale matrices
    cov_i = R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)  # (M, 3, 3)

    delta = mu - mu_merged.unsqueeze(0)         # (M, 3)
    outer = delta.unsqueeze(-1) * delta.unsqueeze(-2)  # (M, 3, 3) outer products

    cov_merged = (w.view(-1, 1, 1) * (cov_i + outer)).sum(dim=0)  # (3, 3)

    # Decompose merged covariance back to scale + rotation
    # SVD: Σ = U D Uᵀ  → R = U, S = sqrt(D)
    try:
        U, D_vals, Vh = torch.linalg.svd(cov_merged)
        # Ensure proper rotation (det = +1)
        if torch.det(U) < 0:
            U[:, -1] = -U[:, -1]
        scale_merged = torch.sqrt(D_vals.clamp(min=1e-10))   # (3,)
        rot_merged   = rotation_matrix_to_quaternion(U.unsqueeze(0)).squeeze(0)  # (4,)
    except Exception:
        # Fallback: use first Gaussian's rotation/scale
        scale_merged = scale[0]
        rot_merged   = rot[0]

    # ---- Merged colour (SH coefficients) ------------------------------------
    fdc   = (w.view(-1, 1, 1) * gaussians._features_dc[indices]).sum(dim=0, keepdim=True)    # (1, 1, 3)
    frest = (w.view(-1, 1, 1) * gaussians._features_rest[indices].mean(dim=0, keepdim=True)).mean(dim=0, keepdim=True)  # rough average

    # ---- Merged opacity (total opacity = weighted sum) -----------------------
    alpha_merged_raw = (w * gaussians._opacity[indices].squeeze(1)).sum()
    alpha_merged_raw = alpha_merged_raw.clamp(-20.0, 20.0)

    return {
        "_xyz":          mu_merged.unsqueeze(0),                         # (1, 3)
        "_features_dc":  fdc,                                            # (1, 1, 3)
        "_features_rest": frest,                                         # (1, 1, 3) – padded below
        "_scaling":      torch.log(scale_merged.clamp(min=1e-10)).unsqueeze(0),  # (1, 3)
        "_rotation":     rot_merged.unsqueeze(0),                        # (1, 4)
        "_opacity":      alpha_merged_raw.reshape(1, 1),                 # (1, 1)
    }


# ---------------------------------------------------------------------------
# HierarchicalLOD
# ---------------------------------------------------------------------------

class HierarchicalLOD:
    """
    Octree-style Hierarchical Level-of-Detail for 3D Gaussian Splatting.

    Levels 0..L-1 are computed via spatial k-means clustering of Gaussian centers.
    Level 0 = finest detail (all Gaussians or finest cluster granularity).
    Level L-1 = coarsest detail (fewest representative Gaussians).

    LOD selection criterion:
        level k selected when camera_dist > base_extent * 2^k

    Usage::

        lod = HierarchicalLOD()
        lod.build(gaussians, levels=4)
        visible = lod.get_visible_gaussians(gaussians, camera)
    """

    def __init__(self) -> None:
        # Per-level: list of (cluster_id → list[gaussian_idx]) mappings
        self._levels: List[List[torch.Tensor]] = []  # level → list of index tensors (one per cluster)
        self._base_extent: float = 1.0
        self._n_levels: int = 0
        # Pre-merged GaussianModels per level (built lazily)
        self._merged_cache: dict[int, GaussianModel] = {}
        self._sh_degree: int = 3

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, gaussians: GaussianModel, levels: int = 4) -> None:
        """
        Cluster Gaussians into `levels` octree-style LOD levels.

        Level 0: individual Gaussians (no clustering).
        Level k (k≥1): k-means with k_clusters = max(1, N // 2^k) clusters.

        Computes base_extent as the diameter of the axis-aligned bounding box.

        Args:
            gaussians: GaussianModel to cluster
            levels:    number of LOD levels (including level 0 = full detail)
        """
        assert levels >= 1, "Need at least 1 level"
        N = gaussians.num_gaussians
        assert N > 0, "GaussianModel has no Gaussians"

        xyz = gaussians.get_xyz.detach()  # (N, 3)

        # Compute scene extent (AABB diagonal)
        bbox_min = xyz.min(dim=0).values
        bbox_max = xyz.max(dim=0).values
        self._base_extent = (bbox_max - bbox_min).norm().item()
        if self._base_extent < 1e-6:
            self._base_extent = 1.0

        self._n_levels = levels
        self._sh_degree = gaussians.max_sh_degree
        self._levels = []
        self._merged_cache = {}

        # Level 0: each Gaussian is its own "cluster"
        level0_clusters = [torch.tensor([i], device=xyz.device) for i in range(N)]
        self._levels.append(level0_clusters)

        # Levels 1..levels-1: progressively coarser k-means
        for lvl in range(1, levels):
            n_clusters = max(1, N // (2 ** lvl))
            if n_clusters >= N:
                # Just copy previous level
                self._levels.append(list(self._levels[-1]))
                continue

            _, assignments = _torch_kmeans(xyz, k=n_clusters)
            clusters = []
            for c in range(n_clusters):
                idxs = (assignments == c).nonzero(as_tuple=False).squeeze(1)
                if idxs.numel() > 0:
                    clusters.append(idxs)
            self._levels.append(clusters)

        print(
            f"[HierarchicalLOD] Built {levels} LOD levels. "
            f"N={N}, base_extent={self._base_extent:.3f}. "
            f"Clusters per level: {[len(l) for l in self._levels]}"
        )

    # ------------------------------------------------------------------
    # LOD selection
    # ------------------------------------------------------------------

    def get_lod_level(self, camera_distance: float, base_level: int = 0) -> int:
        """
        Select LOD level k based on camera distance.

        Mathematical criterion:
            level k selected when camera_dist > base_extent * 2^k

        So:
            k = floor(log2(camera_dist / base_extent))
        clamped to [base_level, n_levels - 1].

        Args:
            camera_distance: distance from camera to scene center (world units)
            base_level:      minimum LOD level to return (default 0 = full detail)

        Returns:
            integer LOD level in [base_level, n_levels - 1]
        """
        if self._n_levels == 0:
            return 0

        if camera_distance <= 0 or self._base_extent <= 0:
            return base_level

        # k = floor(log2(dist / base_extent)) but clamp to valid range
        ratio = camera_distance / self._base_extent
        if ratio <= 1.0:
            level = 0
        else:
            level = int(math.floor(math.log2(ratio)))

        return max(base_level, min(level, self._n_levels - 1))

    # ------------------------------------------------------------------
    # Visible Gaussians
    # ------------------------------------------------------------------

    def get_visible_gaussians(
        self,
        gaussians: GaussianModel,
        camera: Camera,
        target_fps: float = 60.0,
    ) -> GaussianModel:
        """
        Return a GaussianModel subset appropriate for the camera's LOD level.

        The budget heuristic: at target_fps=60, level 0 is used when the
        camera is close; farther cameras automatically receive coarser levels
        (fewer merged Gaussians → faster render).

        If level == 0, returns all Gaussians unchanged.
        Otherwise, returns a merged GaussianModel from the selected level's cache.

        Args:
            gaussians:  source GaussianModel
            camera:     current Camera (provides camera_center)
            target_fps: hint for LOD aggressiveness (currently used for level selection scale)

        Returns:
            GaussianModel (may be a new merged model or the original)
        """
        # Scene center: mean of all Gaussian positions
        with torch.no_grad():
            scene_center = gaussians.get_xyz.mean(dim=0)

        cam_center = camera.camera_center.to(gaussians._xyz.device)
        camera_distance = (cam_center - scene_center).norm().item()

        # Scale LOD aggressiveness: higher target_fps → prefer coarser levels sooner
        fps_scale = math.log2(max(target_fps, 1.0)) / math.log2(60.0)
        adjusted_dist = camera_distance * fps_scale
        level = self.get_lod_level(adjusted_dist)

        if level == 0:
            return gaussians

        return self._get_merged_model(gaussians, level)

    def _get_merged_model(self, gaussians: GaussianModel, level: int) -> GaussianModel:
        """
        Build (and cache) a merged GaussianModel for the given LOD level.
        """
        if level in self._merged_cache:
            return self._merged_cache[level]

        merged = self.merge_gaussians(
            gaussians,
            cluster_list=self._levels[level],
        )
        self._merged_cache[level] = merged
        return merged

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def merge_gaussians(
        self,
        gaussians: GaussianModel,
        gaussian_list: Optional[List[GaussianModel]] = None,
        cluster_list: Optional[List[torch.Tensor]] = None,
    ) -> GaussianModel:
        """
        Merge a list of Gaussian clusters into a single GaussianModel.

        Two usage modes:
          1. Pass `cluster_list`: list of index tensors into `gaussians`.
             Each cluster is merged into one representative Gaussian.
          2. Pass `gaussian_list`: list of GaussianModel objects, each
             treated as a separate cluster (all Gaussians within each model merged).

        In both cases, the mean-of-Gaussian-mixtures approximation is used:
          μ_merged  = Σ wᵢ μᵢ
          Σ_merged  = Σ wᵢ (Σᵢ + δᵢ δᵢᵀ)   where δᵢ = μᵢ - μ̄

        Args:
            gaussians:    source GaussianModel (used in cluster_list mode)
            gaussian_list: list of GaussianModel objects to merge
            cluster_list:  list of index tensors into gaussians

        Returns:
            merged GaussianModel
        """
        if gaussian_list is not None:
            # Merge each model's Gaussians individually, treat each model as a cluster
            merged_parts = []
            for gm in gaussian_list:
                if gm.num_gaussians == 0:
                    continue
                all_idx = torch.arange(gm.num_gaussians, device=gm._xyz.device)
                merged_parts.append(_merge_cluster(gm, all_idx))
            source_sh = gaussian_list[0].max_sh_degree if gaussian_list else gaussians.max_sh_degree
        elif cluster_list is not None:
            merged_parts = []
            for idxs in cluster_list:
                if idxs.numel() == 0:
                    continue
                merged_parts.append(_merge_cluster(gaussians, idxs))
            source_sh = gaussians.max_sh_degree
        else:
            raise ValueError("Either gaussian_list or cluster_list must be provided")

        if not merged_parts:
            return GaussianModel(sh_degree=gaussians.max_sh_degree)

        # Concatenate all merged representatives
        K_rest = gaussians._features_rest.shape[1]  # number of rest SH coefficients

        xyz_all    = torch.cat([p["_xyz"]    for p in merged_parts], dim=0)
        fdc_all    = torch.cat([p["_features_dc"]  for p in merged_parts], dim=0)
        # Pad rest features to correct shape
        frest_list = []
        for p in merged_parts:
            fr = p["_features_rest"]  # (1, m, 3) where m may be 1 after averaging
            # Expand or slice to match K_rest
            if fr.shape[1] < K_rest:
                pad = torch.zeros(fr.shape[0], K_rest - fr.shape[1], 3,
                                  device=fr.device, dtype=fr.dtype)
                fr = torch.cat([fr, pad], dim=1)
            elif fr.shape[1] > K_rest:
                fr = fr[:, :K_rest, :]
            frest_list.append(fr)
        frest_all  = torch.cat(frest_list, dim=0)
        scale_all  = torch.cat([p["_scaling"]  for p in merged_parts], dim=0)
        rot_all    = torch.cat([p["_rotation"] for p in merged_parts], dim=0)
        opacity_all = torch.cat([p["_opacity"] for p in merged_parts], dim=0)

        merged_model = GaussianModel(sh_degree=source_sh)
        merged_model._xyz           = nn.Parameter(xyz_all)
        merged_model._features_dc   = nn.Parameter(fdc_all)
        merged_model._features_rest = nn.Parameter(frest_all)
        merged_model._scaling       = nn.Parameter(scale_all)
        merged_model._rotation      = nn.Parameter(rot_all)
        merged_model._opacity       = nn.Parameter(opacity_all)
        merged_model._init_densification_stats()

        return merged_model

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if self._n_levels == 0:
            return "HierarchicalLOD(not built)"
        cluster_counts = [len(l) for l in self._levels]
        return (
            f"HierarchicalLOD("
            f"levels={self._n_levels}, "
            f"base_extent={self._base_extent:.3f}, "
            f"clusters_per_level={cluster_counts})"
        )
