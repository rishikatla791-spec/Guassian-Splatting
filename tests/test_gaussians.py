"""
Unit tests for GaussianModel.
Run: pytest tests/test_gaussians.py -v
"""
import math
import tempfile
from pathlib import Path
import pytest
import torch
import numpy as np

from gaussian.core.gaussians import GaussianModel


def make_gaussians(n=100, sh_degree=3, device='cpu'):
    """Create a GaussianModel initialized from random point cloud."""
    pts = np.random.randn(n, 3).astype(np.float32)
    colors = np.random.rand(n, 3).astype(np.float32)
    g = GaussianModel(sh_degree=sh_degree)
    g.init_from_pointcloud(pts, colors)
    return g


class TestGaussianModel:

    def test_init_shapes(self):
        N = 200
        g = make_gaussians(N)
        assert g._xyz.shape == (N, 3)
        assert g._features_dc.shape == (N, 1, 3)
        assert g._features_rest.shape == (N, 15, 3)  # 16-1=15 for SH degree 3
        assert g._opacity.shape == (N, 1)
        assert g._scaling.shape == (N, 3)
        assert g._rotation.shape == (N, 4)

    def test_opacity_in_range(self):
        g = make_gaussians(100)
        opa = g.get_opacity
        assert opa.shape == (100, 1)
        assert (opa >= 0.0).all() and (opa <= 1.0).all()

    def test_scale_positive(self):
        g = make_gaussians(100)
        scales = g.get_scaling
        assert (scales > 0.0).all(), "All scales must be positive (exp activation)"

    def test_rotation_normalized(self):
        g = make_gaussians(100)
        rots = g.get_rotation
        norms = rots.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(100), atol=1e-6), \
            f"Rotations not normalized: min={norms.min():.6f}, max={norms.max():.6f}"

    def test_covariance_shape(self):
        g = make_gaussians(50)
        cov = g.get_covariance(scale_modifier=1.0)
        assert cov.shape == (50, 6)

    def test_features_shape(self):
        g = make_gaussians(50, sh_degree=3)
        feats = g.get_features
        assert feats.shape == (50, 16, 3)  # (sh_degree+1)² = 16

    def test_num_gaussians(self):
        N = 77
        g = make_gaussians(N)
        assert g.num_gaussians == N

    def test_save_load_ply(self):
        """Save to PLY and reload — all parameters must match."""
        g1 = make_gaussians(50)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.ply"
            g1.save_ply(path)
            g2 = GaussianModel(sh_degree=3)
            g2.load_ply(path)

        assert g1.num_gaussians == g2.num_gaussians
        assert torch.allclose(g1._xyz.detach(), g2._xyz.detach(), atol=1e-5)
        assert torch.allclose(g1._opacity.detach(), g2._opacity.detach(), atol=1e-5)
        assert torch.allclose(g1._scaling.detach(), g2._scaling.detach(), atol=1e-5)

    def test_densification_stats_update(self):
        """After add_densification_stats, denom should increase."""
        g = make_gaussians(100)
        N = g.num_gaussians

        vp = torch.zeros(N, 3, requires_grad=True)
        # Simulate a gradient on viewspace_points
        loss = vp[:, :2].sum()
        loss.backward()

        mask = torch.ones(N, dtype=torch.bool)
        initial_denom = g.denom.clone()
        g.add_densification_stats(vp, mask)
        assert (g.denom >= initial_denom).all(), "denom should never decrease"
        assert (g.denom[mask] > initial_denom[mask]).all(), "denom should increase for visible"

    def test_sh_degree_progression(self):
        g = make_gaussians(10)
        assert g.active_sh_degree == 0
        g.oneupSHdegree()
        assert g.active_sh_degree == 1
        g.oneupSHdegree()
        assert g.active_sh_degree == 2
        g.oneupSHdegree()
        assert g.active_sh_degree == 3
        g.oneupSHdegree()
        assert g.active_sh_degree == 3, "Should not exceed max_sh_degree"
