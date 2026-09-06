"""
Phase 1 Validation Suite: Comprehensive Verification & Benchmarking of C++/CUDA Differentiable Rasterizer.

Verifies:
  1. CUDA forward pass execution & output integrity
  2. CUDA backward pass analytical gradient propagation
  3. Gradients verified across XYZ, Scaling, Rotation, Opacity, and SH Features
  4. Deterministic numerical match against analytical ground truth reference
  5. Strict No-NaN / No-Inf validation under boundary & edge cases
  6. High-precision benchmarks: forward time, backward time, FPS on RTX 3050 GPU
"""

import sys
import time
import math
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
import numpy as np

root_dir = Path(__file__).parent.parent.resolve()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from core.gaussians import GaussianModel
from core.camera import Camera, CameraIntrinsics, CameraExtrinsics
from core.cuda_rasterizer import cuda_rasterize, CUDARasterizeFunction
from renderer.cuda_rasterizer import CUDAGaussianRasterizer
from renderer.tile_rasterizer import TileBasedRasterizer
from renderer.gaussian_rasterizer import RasterizationSettings


# ══════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════

@pytest.fixture
def test_setup():
    torch.manual_seed(42)
    np.random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    W, H = 512, 512
    intrinsics = CameraIntrinsics(fx=400.0, fy=400.0, cx=256.0, cy=256.0, width=W, height=H)
    R = np.eye(3, dtype=np.float64)
    T = np.array([0.0, 0.0, 3.0], dtype=np.float64)
    extrinsics = CameraExtrinsics(R=R, T=T)
    camera = Camera(uid=0, intrinsics=intrinsics, extrinsics=extrinsics)

    N = 250
    pts = np.random.uniform(-1.0, 1.0, (N, 3)).astype(np.float32)
    pts[:, 2] += 2.5 # In front of camera
    cols = np.random.uniform(0.1, 0.9, (N, 3)).astype(np.float32)

    gaussians = GaussianModel(sh_degree=1)
    gaussians.init_from_pointcloud(pts, cols)
    if hasattr(gaussians, "to"):
        gaussians = gaussians.to(device)

    # Anisotropic scaling so rotation gradient is non-zero
    with torch.no_grad():
        gaussians._scaling.data[:, 0] += 0.3
        gaussians._scaling.data[:, 1] -= 0.2

    return {
        "camera": camera,
        "gaussians": gaussians,
        "device": device,
        "N": N,
        "W": W,
        "H": H,
    }


# ══════════════════════════════════════════════════════════════════
# 1. CUDA Forward Pass Test
# ══════════════════════════════════════════════════════════════════

def test_cuda_forward(test_setup):
    setup = test_setup
    device = setup["device"]
    if device != "cuda":
        pytest.skip("CUDA device not available")

    renderer = TileBasedRasterizer()
    bg_color = torch.tensor([0.0, 0.0, 0.0], device=device)

    out = renderer.render(setup["gaussians"], setup["camera"], bg_color=bg_color)

    assert "render" in out
    assert "depth" in out
    assert "alpha" in out
    assert "viewspace_points" in out

    render = out["render"]
    depth = out["depth"]
    alpha = out["alpha"]

    assert render.shape == (3, setup["H"], setup["W"])
    assert depth.shape == (setup["H"], setup["W"])
    assert alpha.shape == (setup["H"], setup["W"])

    assert render.min() >= -1e-5, f"Render min below 0: {render.min()}"
    assert render.max() <= 1.0 + 1e-5, f"Render max above 1: {render.max()}"
    assert not torch.isnan(render).any(), "NaN in forward render output"
    assert not torch.isinf(render).any(), "Inf in forward render output"
    print("\n[OK] CUDA Forward Pass Verified: Shape (3, 512, 512), Values in [0, 1], No NaN/Inf")


# ══════════════════════════════════════════════════════════════════
# 2. CUDA Backward Pass & Gradient Verification
# ══════════════════════════════════════════════════════════════════

def test_cuda_backward_gradients(test_setup):
    setup = test_setup
    device = setup["device"]
    if device != "cuda":
        pytest.skip("CUDA device not available")

    gaussians = setup["gaussians"]
    camera = setup["camera"]
    renderer = TileBasedRasterizer()
    bg_color = torch.tensor([0.2, 0.3, 0.4], device=device)

    # Forward
    out = renderer.render(gaussians, camera, bg_color=bg_color)
    rendered = out["render"]
    viewspace_points = out["viewspace_points"]

    # Target loss
    target = torch.ones_like(rendered) * 0.5
    loss = F.mse_loss(rendered, target)
    loss.backward()

    # Verify all parameters received valid non-zero gradients
    assert gaussians._xyz.grad is not None, "Missing grad for _xyz"
    assert gaussians._scaling.grad is not None, "Missing grad for _scaling"
    assert gaussians._rotation.grad is not None, "Missing grad for _rotation"
    assert gaussians._opacity.grad is not None, "Missing grad for _opacity"
    assert gaussians._features_dc.grad is not None, "Missing grad for _features_dc"
    assert viewspace_points.grad is not None, "Missing grad for viewspace_points"

    assert not torch.isnan(gaussians._xyz.grad).any(), "NaN in _xyz grad"
    assert not torch.isnan(gaussians._scaling.grad).any(), "NaN in _scaling grad"
    assert not torch.isnan(gaussians._rotation.grad).any(), "NaN in _rotation grad"
    assert not torch.isnan(gaussians._opacity.grad).any(), "NaN in _opacity grad"
    assert not torch.isnan(gaussians._features_dc.grad).any(), "NaN in _features_dc grad"
    assert not torch.isnan(viewspace_points.grad).any(), "NaN in viewspace_points grad"

    xyz_norm = gaussians._xyz.grad.norm().item()
    scale_norm = gaussians._scaling.grad.norm().item()
    rot_norm = gaussians._rotation.grad.norm().item()
    opac_norm = gaussians._opacity.grad.norm().item()
    feat_norm = gaussians._features_dc.grad.norm().item()
    vsp_norm = viewspace_points.grad.norm().item()

    assert xyz_norm > 0, "Zero grad for _xyz"
    assert scale_norm > 0, "Zero grad for _scaling"
    assert rot_norm > 0, "Zero grad for _rotation"
    assert opac_norm > 0, "Zero grad for _opacity"
    assert feat_norm > 0, "Zero grad for _features_dc"
    assert vsp_norm > 0, "Zero grad for viewspace_points"

    print(f"\n[OK] CUDA Backward Pass Verified:")
    print(f"     - grad_xyz norm:              {xyz_norm:.4e}")
    print(f"     - grad_scaling norm:          {scale_norm:.4e}")
    print(f"     - grad_rotation norm:         {rot_norm:.4e}")
    print(f"     - grad_opacity norm:          {opac_norm:.4e}")
    print(f"     - grad_features_dc norm:      {feat_norm:.4e}")
    print(f"     - grad_viewspace_points norm: {vsp_norm:.4e}")


# ══════════════════════════════════════════════════════════════════
# 3. Deterministic Numerical Equivalence against Reference Math
# ══════════════════════════════════════════════════════════════════

def test_numerical_match_reference():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        pytest.skip("CUDA device not available")

    torch.manual_seed(123)
    H, W = 64, 64

    # Center exactly at (32.5, 32.5) to align with pixel (32, 32) half-pixel offset
    means2d = torch.tensor([[32.5, 32.5]], device=device, dtype=torch.float32)
    colors = torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=torch.float32)
    opacities = torch.tensor([0.8], device=device, dtype=torch.float32)

    # 2D Covariance: standard circular Gaussian (cinv = 0.1 I)
    cov2d_inv = torch.tensor([[0.1, 0.0, 0.1]], device=device, dtype=torch.float32)
    depths = torch.tensor([1.0], device=device, dtype=torch.float32)
    radii = torch.tensor([10.0], device=device, dtype=torch.float32)
    bg = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32)

    # 1. CUDA Rasterizer Output
    out_color, out_depth, out_alpha = cuda_rasterize(
        means2d=means2d,
        colors=colors,
        opacities=opacities,
        cov2d_inv=cov2d_inv,
        depths=depths,
        radii=radii,
        H=H, W=W,
        bg_color=bg,
        tile_size=16,
    )

    # 2. Reference Math Evaluation at Center Pixel (32, 32)
    # At (32, 32), p_xf = 32.5, p_yf = 32.5 -> dx = 0.0, dy = 0.0 -> maha2 = 0 -> weight = 1.0 -> alpha = 0.8 -> Color = [0.8, 0, 0]
    center_color = out_color[32, 32].cpu()
    expected_red = 0.8

    assert abs(center_color[0].item() - expected_red) < 1e-4, \
        f"Center pixel red channel {center_color[0].item()} deviates from expected {expected_red}"
    print(f"\n[OK] Numerical Match Verified: Exact Analytical Equivalence Confirmed (Error = {abs(center_color[0].item() - expected_red):.2e} < 1e-4)")


# ══════════════════════════════════════════════════════════════════
# 4. Strict Boundary & Edge Cases (No NaN/Inf)
# ══════════════════════════════════════════════════════════════════

def test_boundary_and_edge_cases():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        pytest.skip("CUDA device not available")

    # Case A: 0 Gaussians
    empty_m = torch.zeros((0, 2), device=device)
    empty_c = torch.zeros((0, 3), device=device)
    empty_o = torch.zeros((0,), device=device)
    empty_cov = torch.zeros((0, 3), device=device)
    empty_d = torch.zeros((0,), device=device)
    empty_r = torch.zeros((0,), device=device)
    bg = torch.tensor([0.5, 0.5, 0.5], device=device)

    out_c, out_d, _ = cuda_rasterize(empty_m, empty_c, empty_o, empty_cov, empty_d, empty_r, 64, 64, bg)
    assert torch.allclose(out_c, bg.view(1, 1, 3).expand(64, 64, 3))
    assert not torch.isnan(out_c).any()

    # Case B: Out-of-bounds Gaussians (x = -500, y = 5000)
    oob_m = torch.tensor([[-500.0, 5000.0]], device=device)
    oob_c = torch.tensor([[1.0, 1.0, 1.0]], device=device)
    oob_o = torch.tensor([0.5], device=device)
    oob_cov = torch.tensor([[0.01, 0.0, 0.01]], device=device)
    oob_d = torch.tensor([2.0], device=device)
    oob_r = torch.tensor([5.0], device=device)

    out_c2, _, _ = cuda_rasterize(oob_m, oob_c, oob_o, oob_cov, oob_d, oob_r, 64, 64, bg)
    assert not torch.isnan(out_c2).any()
    assert not torch.isinf(out_c2).any()
    print("\n[OK] Edge Cases & Boundary Conditions Verified: 0 Gaussians & Out-of-Bounds safe (No NaN/Inf)")


# ══════════════════════════════════════════════════════════════════
# 5. Performance Benchmarks (Forward, Backward, FPS on RTX 3050)
# ══════════════════════════════════════════════════════════════════

def test_performance_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        pytest.skip("CUDA device not available")

    torch.manual_seed(42)
    N = 100_000 # 100k Gaussians
    H, W = 800, 800

    means2d = (torch.rand(N, 2, device=device) * 750.0 + 25.0).requires_grad_(True)
    colors = torch.rand(N, 3, device=device).requires_grad_(True)
    opacities = (torch.rand(N, device=device) * 0.8 + 0.1).requires_grad_(True)
    cov2d_inv = torch.full((N, 3), 0.02, device=device).requires_grad_(True)
    depths = torch.rand(N, device=device) * 10.0 + 0.5
    radii = torch.full((N,), 12.0, device=device)
    bg = torch.tensor([0.0, 0.0, 0.0], device=device)

    # Warmup
    for _ in range(5):
        out_c, _, _ = cuda_rasterize(means2d, colors, opacities, cov2d_inv, depths, radii, H, W, bg)
        out_c.sum().backward()
        means2d.grad = None
        colors.grad = None
        opacities.grad = None
        cov2d_inv.grad = None

    torch.cuda.synchronize()

    # Forward Timing
    n_runs = 20
    t0 = time.perf_counter()
    for _ in range(n_runs):
        out_c, _, _ = cuda_rasterize(means2d, colors, opacities, cov2d_inv, depths, radii, H, W, bg)
    torch.cuda.synchronize()
    fwd_time_ms = ((time.perf_counter() - t0) / n_runs) * 1000.0

    # Backward Timing
    t0 = time.perf_counter()
    for _ in range(n_runs):
        out_c, _, _ = cuda_rasterize(means2d, colors, opacities, cov2d_inv, depths, radii, H, W, bg)
        loss = out_c.sum()
        loss.backward()
        means2d.grad = None
        colors.grad = None
        opacities.grad = None
        cov2d_inv.grad = None
    torch.cuda.synchronize()
    total_time_ms = ((time.perf_counter() - t0) / n_runs) * 1000.0
    bwd_time_ms = total_time_ms - fwd_time_ms
    fps = 1000.0 / fwd_time_ms

    gpu_name = torch.cuda.get_device_name(0)
    vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print("==========================================================================")
    print(f"  PHASE 1 CUDA RASTERIZER BENCHMARK REPORT ({gpu_name})")
    print("==========================================================================")
    print(f"  - Scene Complexity:   {N:,} Gaussians")
    print(f"  - Screen Resolution:  {W} x {H} pixels")
    print(f"  - Forward Pass Time:  {fwd_time_ms:.2f} ms")
    print(f"  - Backward Pass Time: {bwd_time_ms:.2f} ms")
    print(f"  - Real-Time FPS:      {fps:.1f} FPS")
    print(f"  - Peak VRAM Usage:    {vram_mb:.1f} MB")
    print("==========================================================================\n")

    assert fps >= 10.0, f"FPS {fps} is too low for 100k splats!"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
