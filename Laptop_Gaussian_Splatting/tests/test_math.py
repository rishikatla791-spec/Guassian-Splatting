"""
Unit tests for core mathematical utilities.
Run: pytest tests/test_math.py -v
"""
import math
import pytest
import torch
import numpy as np

from gaussian.core.math_utils import (
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    build_covariance_3d,
    build_covariance_2d,
    invert_cov2d,
    compute_radius_from_cov2d,
    project_points,
    ndc_to_screen,
)
from gaussian.core.sh import eval_sh, RGB2SH, SH2RGB


# ---------------------------------------------------------------------------
# Quaternion tests
# ---------------------------------------------------------------------------

class TestQuaternion:

    def test_identity_quaternion_gives_identity_R(self):
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # w=1, x=y=z=0
        R = quaternion_to_rotation_matrix(q)
        assert R.shape == (1, 3, 3)
        assert torch.allclose(R[0], torch.eye(3), atol=1e-6)

    def test_quaternion_roundtrip(self):
        """R → q → R should recover original R (up to sign)."""
        torch.manual_seed(42)
        for _ in range(20):
            # Random rotation via random unit quaternion
            q0 = torch.randn(1, 4)
            q0 = q0 / q0.norm()
            R0 = quaternion_to_rotation_matrix(q0)
            q1 = rotation_matrix_to_quaternion(R0)
            R1 = quaternion_to_rotation_matrix(q1)
            assert torch.allclose(R0, R1, atol=1e-5), f"Roundtrip failed: max err={( R0-R1).abs().max()}"

    def test_rotation_matrix_orthogonal(self):
        """R @ Rᵀ = I."""
        q = torch.randn(50, 4)
        q = q / q.norm(dim=-1, keepdim=True)
        R = quaternion_to_rotation_matrix(q)  # (50, 3, 3)
        I = torch.bmm(R, R.transpose(-1, -2))
        assert torch.allclose(I, torch.eye(3).unsqueeze(0).expand(50, 3, 3), atol=1e-5)

    def test_rotation_matrix_det_one(self):
        """det(R) = 1."""
        q = torch.randn(50, 4)
        q = q / q.norm(dim=-1, keepdim=True)
        R = quaternion_to_rotation_matrix(q)
        dets = torch.linalg.det(R)
        assert torch.allclose(dets, torch.ones(50), atol=1e-5)

    def test_90_degree_rotation(self):
        """90° rotation around z-axis: q = [cos45°, 0, 0, sin45°]."""
        angle = math.pi / 2.0
        q = torch.tensor([[math.cos(angle/2), 0.0, 0.0, math.sin(angle/2)]])
        R = quaternion_to_rotation_matrix(q)[0]
        x_rotated = R @ torch.tensor([1.0, 0.0, 0.0])
        assert torch.allclose(x_rotated, torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)


# ---------------------------------------------------------------------------
# Covariance tests
# ---------------------------------------------------------------------------

class TestCovariance:

    def test_covariance_3d_shape(self):
        N = 100
        scales = torch.zeros(N, 3)
        rots = torch.zeros(N, 4); rots[:, 0] = 1.0
        cov = build_covariance_3d(scales, 1.0, rots)
        assert cov.shape == (N, 6)

    def test_covariance_3d_psd(self):
        """
        Σ = R S Sᵀ Rᵀ must be positive semi-definite.
        PSD: all eigenvalues ≥ 0.
        """
        N = 50
        scales = torch.randn(N, 3) * 0.5
        rots = torch.randn(N, 4)
        rots = rots / rots.norm(dim=-1, keepdim=True)
        cov6 = build_covariance_3d(scales, 1.0, rots)

        # Reconstruct full 3×3 symmetric matrix
        from gaussian.core.math_utils import _upper_tri_to_matrix
        cov3x3 = _upper_tri_to_matrix(cov6)  # (N, 3, 3)

        eigvals = torch.linalg.eigvalsh(cov3x3)  # (N, 3) sorted ascending
        assert (eigvals >= -1e-5).all(), f"Found negative eigenvalue: {eigvals.min()}"

    def test_cov2d_projection_psd(self):
        """Projected 2D covariance must be positive definite."""
        N = 20
        means3d = torch.randn(N, 3)
        means3d[:, 2] += 5.0  # put in front of camera

        scales = torch.zeros(N, 3)  # unit scales
        rots = torch.zeros(N, 4); rots[:, 0] = 1.0
        cov6 = build_covariance_3d(scales, 1.0, rots)

        viewmat = torch.eye(4)
        projmat = torch.tensor([
            [1.0, 0, 0, 0],
            [0, 1.0, 0, 0],
            [0, 0, -1.01, -0.2],
            [0, 0, -1.0, 0],
        ])
        cov2d, _ = build_covariance_2d(
            means3d, cov6, viewmat,
            fovx=math.pi/3, fovy=math.pi/3,
            image_width=800, image_height=600
        )
        # Check [a, b, c] → [[a,b],[b,c]] is PSD: a>0, det=ac-b²>0
        a, b, c = cov2d[:, 0], cov2d[:, 1], cov2d[:, 2]
        assert (a > 0).all(), "a must be positive"
        det = a * c - b * b
        assert (det > -1e-5).all(), f"det must be >= 0, got {det.min()}"

    def test_invert_cov2d_identity(self):
        """cov2d @ cov2d_inv ≈ I."""
        cov2d = torch.tensor([[2.0, 0.5, 1.5],
                               [3.0, 1.0, 2.0]])
        inv, det = invert_cov2d(cov2d)
        for i in range(cov2d.shape[0]):
            a, b, c = cov2d[i, 0], cov2d[i, 1], cov2d[i, 2]
            M = torch.tensor([[a, b], [b, c]])
            result = M @ inv[i]
            assert torch.allclose(result, torch.eye(2), atol=1e-5), \
                f"Inverse failed for row {i}: {result}"


# ---------------------------------------------------------------------------
# Spherical harmonics tests
# ---------------------------------------------------------------------------

class TestSphericalHarmonics:

    def test_band0_constant(self):
        """SH degree 0 should give same value for any direction."""
        N = 10
        sh = torch.ones(N, 1, 3)  # all-ones DC coefficients
        dirs = torch.randn(N, 3)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        colors = eval_sh(0, sh, dirs)
        # All should be the same (DC only)
        assert torch.allclose(colors, colors[0:1].expand_as(colors), atol=1e-6)

    def test_rgb2sh_roundtrip(self):
        """RGB → SH DC → RGB roundtrip must be identity."""
        rgb = torch.rand(100, 3)
        sh_dc = RGB2SH(rgb)
        rgb_back = SH2RGB(sh_dc)
        assert torch.allclose(rgb, rgb_back, atol=1e-6)

    def test_sh_degree_3_shape(self):
        N = 50
        sh = torch.randn(N, 16, 3)
        dirs = torch.randn(N, 3)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        colors = eval_sh(3, sh, dirs)
        assert colors.shape == (N, 3)

    def test_sh_orthogonality_monte_carlo(self):
        """
        Monte Carlo test: integrate sh(d) over unit sphere ≈ DC contribution only.
        For unit sphere with uniform distribution, integral of Y_l^m = 0 for l > 0.
        So ∫ eval_sh(d) d\Omega ≈ C0 * sh[:,0] * 4π  (band-0 contribution).
        """
        N_samples = 10000
        N_gauss = 5
        torch.manual_seed(0)

        sh = torch.zeros(N_gauss, 16, 3)
        sh[:, 0] = 1.0  # only DC component

        # Uniform random directions on sphere
        theta = torch.acos(1 - 2 * torch.rand(N_samples))
        phi = 2 * math.pi * torch.rand(N_samples)
        dirs = torch.stack([
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta),
        ], dim=-1)  # (N_samples, 3)

        # Evaluate SH for first Gaussian at all directions
        sh_broadcast = sh[0:1].expand(N_samples, 16, 3)
        vals = eval_sh(3, sh_broadcast, dirs)  # (N_samples, 3)
        mean_val = vals.mean(dim=0)  # (3,) should be ≈ C0

        from gaussian.core.sh import C0
        expected = C0 * torch.ones(3)
        assert torch.allclose(mean_val, expected, atol=0.02), \
            f"SH Monte Carlo: got {mean_val}, expected {expected}"


# ---------------------------------------------------------------------------
# Projection tests
# ---------------------------------------------------------------------------

class TestProjection:

    def test_project_points_in_front(self):
        """Points at z=5 (in front of camera) should have positive depth."""
        points = torch.tensor([[0.0, 0.0, 5.0]])
        view = torch.eye(4)
        proj = torch.tensor([
            [1.0, 0, 0, 0],
            [0, 1.0, 0, 0],
            [0, 0, -1.01, -0.2],
            [0, 0, -1.0, 0],
        ])
        ndc, depth = project_points(points, view, proj)
        assert depth[0] > 0, "Depth should be positive for in-front points"

    def test_ndc_to_screen(self):
        """NDC origin should map to image center."""
        ndc = torch.tensor([[0.0, 0.0, 0.5]])
        screen = ndc_to_screen(ndc, width=800, height=600)
        assert abs(screen[0, 0].item() - 400.0) < 1.0
        assert abs(screen[0, 1].item() - 300.0) < 1.0

    def test_radius_positive(self):
        """compute_radius_from_cov2d should return positive integers."""
        cov2d = torch.tensor([[1.0, 0.1, 0.8], [2.0, 0.5, 1.5]])
        radii = compute_radius_from_cov2d(cov2d)
        assert (radii > 0).all()
        assert radii.dtype == torch.int64
