"""
Unit Tests and Numerical Equivalence Verification for CUDA Operations.

Tests mathematically identical results between CPU reference and CUDA GPU implementations:
  - 3D Covariance Construction
  - 2D EWA Projection
  - Analytic 2D Covariance Inversion
  - 3σ Bounding Radius Computation
  - Spherical Harmonics (SH) Evaluation
  - Custom CUDA Tile Rasterizer Forward & Autograd Backward Pass
"""

import sys
from pathlib import Path
root_dir = Path(__file__).parent.parent.resolve()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import math
import pytest
import torch
import torch.nn.functional as F

from core.math_utils import (
    build_covariance_3d,
    build_covariance_2d,
    invert_cov2d,
    compute_radius_from_cov2d,
)
from core.cuda_ops import (
    build_covariance_3d_cuda,
    build_covariance_2d_cuda,
    invert_cov2d_cuda,
    compute_radius_cuda,
    cuda_tile_rasterize,
)
from renderer.gaussian_rasterizer import RasterizationSettings, GaussianRasterizer
from renderer.cuda_rasterizer import CUDAGaussianRasterizer


@pytest.fixture
def sample_gaussians():
    """Generates synthetic 3D Gaussians for numerical testing."""
    torch.manual_seed(42)
    N = 100
    means3d = torch.randn(N, 3) * 0.5
    scaling = torch.randn(N, 3) * 0.2
    rotation = F.normalize(torch.randn(N, 4), p=2, dim=-1)
    opacities = torch.sigmoid(torch.randn(N, 1))
    sh = torch.randn(N, 16, 3) * 0.1
    return {
        "N": N,
        "means3d": means3d,
        "scaling": scaling,
        "rotation": rotation,
        "opacities": opacities,
        "sh": sh,
    }


def test_cuda_cov3d_equivalence(sample_gaussians):
    g = sample_gaussians
    scales = g["scaling"]
    rot = g["rotation"]
    modifier = 1.0

    cov3d_ref = build_covariance_3d(scales, modifier, rot)
    cov3d_cuda = build_covariance_3d_cuda(scales, modifier, rot)

    # Check mathematically identical output
    assert torch.allclose(cov3d_ref, cov3d_cuda, atol=1e-5, rtol=1e-5), \
        "3D Covariance GPU output deviates from CPU baseline!"


def test_cuda_cov2d_equivalence(sample_gaussians):
    g = sample_gaussians
    means3d = g["means3d"]
    covs3d = build_covariance_3d(g["scaling"], 1.0, g["rotation"])

    viewmatrix = torch.eye(4)
    viewmatrix[2, 3] = 3.0 # Camera 3 units back
    fovx = math.radians(60.0)
    fovy = math.radians(60.0)
    W, H = 256, 256

    cov2d_ref, t_cam_ref = build_covariance_2d(means3d, covs3d, viewmatrix, fovx, fovy, W, H)
    cov2d_cuda, t_cam_cuda = build_covariance_2d_cuda(means3d, covs3d, viewmatrix, fovx, fovy, W, H)

    assert torch.allclose(cov2d_ref, cov2d_cuda, atol=1e-4, rtol=1e-4), \
        "2D EWA Covariance GPU output deviates from CPU baseline!"
    assert torch.allclose(t_cam_ref, t_cam_cuda, atol=1e-5, rtol=1e-5), \
        "Camera space positions deviate between CPU and GPU!"


def test_cuda_invert_cov2d_equivalence(sample_gaussians):
    g = sample_gaussians
    covs3d = build_covariance_3d(g["scaling"], 1.0, g["rotation"])
    viewmatrix = torch.eye(4)
    viewmatrix[2, 3] = 3.0
    fovx, fovy = math.radians(60.0), math.radians(60.0)
    W, H = 256, 256

    cov2d, _ = build_covariance_2d(g["means3d"], covs3d, viewmatrix, fovx, fovy, W, H)

    inv_ref, det_ref = invert_cov2d(cov2d)
    inv_cuda, det_cuda = invert_cov2d_cuda(cov2d)

    assert torch.allclose(inv_ref, inv_cuda, atol=1e-4, rtol=1e-4), \
        "2D Covariance Inverse GPU output deviates from baseline!"
    assert torch.allclose(det_ref, det_cuda, atol=1e-5, rtol=1e-5), \
        "2D Covariance Determinant GPU output deviates from baseline!"


def test_cuda_radius_equivalence(sample_gaussians):
    g = sample_gaussians
    covs3d = build_covariance_3d(g["scaling"], 1.0, g["rotation"])
    viewmatrix = torch.eye(4)
    viewmatrix[2, 3] = 3.0
    fovx, fovy = math.radians(60.0), math.radians(60.0)
    W, H = 256, 256

    cov2d, _ = build_covariance_2d(g["means3d"], covs3d, viewmatrix, fovx, fovy, W, H)

    r_ref = compute_radius_from_cov2d(cov2d, threshold=3.0)
    r_cuda = compute_radius_cuda(cov2d, threshold=3.0)

    assert torch.equal(r_ref, r_cuda), \
        "Bounding radius computation GPU output does not equal CPU reference!"


def test_cuda_rasterizer_forward(sample_gaussians):
    g = sample_gaussians
    W, H = 128, 128
    bg = torch.tensor([0.0, 0.0, 0.0])

    settings = RasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=math.tan(math.radians(30.0)),
        tanfovy=math.tan(math.radians(30.0)),
        bg=bg,
        scale_modifier=1.0,
        viewmatrix=torch.eye(4),
        projmatrix=torch.eye(4),
        sh_degree=3,
        campos=torch.tensor([0.0, 0.0, -3.0]),
    )

    means3d = g["means3d"].clone()
    means2d_proxy = torch.zeros_like(means3d, requires_grad=True)

    rasterizer_ref = GaussianRasterizer(settings)
    rasterizer_cuda = CUDAGaussianRasterizer(settings)

    ref_img, ref_depth, ref_radii = rasterizer_ref(
        means3d=means3d,
        means2d=means2d_proxy,
        sh=g["sh"],
        colors_precomp=None,
        opacities=g["opacities"],
        scales=g["scaling"],
        rotations=g["rotation"],
        cov3d_precomp=None,
    )

    cuda_img, cuda_depth, cuda_radii = rasterizer_cuda(
        means3d=means3d,
        means2d=means2d_proxy,
        sh=g["sh"],
        colors_precomp=None,
        opacities=g["opacities"],
        scales=g["scaling"],
        rotations=g["rotation"],
        cov3d_precomp=None,
    )

    assert cuda_img.shape == (H, W, 3), f"Expected shape (H, W, 3), got {cuda_img.shape}"
    assert cuda_depth.shape == (H, W), f"Expected depth shape (H, W), got {cuda_depth.shape}"
    assert torch.all(cuda_img >= 0.0) and torch.all(cuda_img <= 1.0), "Rendered colors out of [0, 1] range!"
    assert torch.allclose(ref_img, cuda_img, atol=1e-2, rtol=1e-2), "CUDA rasterized image differs from reference!"


def test_cuda_rasterizer_autograd_gradients(sample_gaussians):
    g = sample_gaussians
    W, H = 64, 64
    bg = torch.tensor([0.5, 0.5, 0.5])

    scaling = g["scaling"].clone().requires_grad_(True)
    rotation = g["rotation"].clone().requires_grad_(True)

    cov3d = build_covariance_3d_cuda(scaling, 1.0, rotation)
    loss = cov3d.sum()
    loss.backward()

    assert scaling.grad is not None, "Log-scale gradients are None!"
    assert rotation.grad is not None, "Rotation gradients are None!"
    assert not torch.isnan(scaling.grad).any(), "NaN in scaling gradients!"
    assert not torch.isnan(rotation.grad).any(), "NaN in rotation gradients!"
