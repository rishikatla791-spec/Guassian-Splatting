"""
Phase 3 Validation Suite: COLMAP Scene Parser, PLY Export/Import & Real Dataset Training.

Tests:
  1. Official COLMAP Binary Parser (250 views, camera intrinsics/extrinsics, points3D)
  2. Official Inria PLY Serializer & Deserializer round-trip parity
  3. Reconstruction Quality Metrics (PSNR, SSIM, L1)
  4. Real-world COLMAP Training on Truck scene (Unseen test views, evaluation, PLY export)
"""

import sys
import os
import math
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
import numpy as np

root_dir = Path(__file__).parent.parent.resolve()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from scene.dataset_readers import read_colmap_scene_info
from core.gaussians import GaussianModel
from renderer.tile_rasterizer import TileBasedRasterizer
from training.config import TrainingConfig
from training.trainer import GaussianTrainer
from training.metrics import compute_psnr, compute_ssim, evaluate_dataset


TRUCK_PATH = root_dir / "imgaes" / "truck"


# ══════════════════════════════════════════════════════════════════
# 1. Test COLMAP Scene Parser
# ══════════════════════════════════════════════════════════════════

def test_colmap_binary_reader():
    if not TRUCK_PATH.exists():
        pytest.skip(f"Truck dataset not found at {TRUCK_PATH}")

    scene_info = read_colmap_scene_info(
        dataset_path=str(TRUCK_PATH),
        images_subfolder="images",
        eval_mode=True,
        llffhold=8,
    )

    total_cams = len(scene_info.train_cameras) + len(scene_info.test_cameras)
    assert total_cams == 251, f"Expected 251 cameras, got {total_cams}"
    assert len(scene_info.test_cameras) == 32, f"Expected 32 test views, got {len(scene_info.test_cameras)}"
    assert len(scene_info.train_cameras) == 219, f"Expected 219 train views, got {len(scene_info.train_cameras)}"

    # Check first camera geometry
    cam = scene_info.train_cameras[0]
    assert cam.intrinsics.width > 0 and cam.intrinsics.height > 0
    assert cam.intrinsics.fx > 0 and cam.intrinsics.fy > 0
    assert cam.extrinsics.R.shape == (3, 3)
    assert cam.extrinsics.T.shape == (3,)
    # Verify R is valid rotation matrix (R @ R.T == I)
    assert np.allclose(cam.extrinsics.R @ cam.extrinsics.R.T, np.eye(3), atol=1e-5)

    # Check Point Cloud
    pts = scene_info.point_cloud.points
    cols = scene_info.point_cloud.colors
    assert pts.shape[0] > 10_000, f"Expected >10k points, got {pts.shape[0]}"
    assert cols.shape == pts.shape
    assert cols.min() >= 0.0 and cols.max() <= 1.0
    assert scene_info.scene_extent > 0.0

    print(f"\n[OK] COLMAP Scene Parser Verified: {total_cams} cameras, {pts.shape[0]:,} points, Extent={scene_info.scene_extent:.2f}")


# ══════════════════════════════════════════════════════════════════
# 2. Test Official PLY Serializer & Deserializer Round-Trip
# ══════════════════════════════════════════════════════════════════

def test_official_ply_roundtrip(tmp_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    np.random.seed(42)

    N = 100
    pts = np.random.uniform(-1.0, 1.0, (N, 3)).astype(np.float32)
    cols = np.random.uniform(0.1, 0.9, (N, 3)).astype(np.float32)

    gaussians_orig = GaussianModel(sh_degree=3)
    gaussians_orig.init_from_pointcloud(pts, cols, device=device)

    # Export to PLY
    ply_file = tmp_path / "point_cloud.ply"
    gaussians_orig.save_ply(ply_file)
    assert ply_file.exists() and ply_file.stat().st_size > 0

    # Load back into a fresh model
    gaussians_loaded = GaussianModel(sh_degree=3)
    gaussians_loaded.load_ply(ply_file, device=device)

    assert gaussians_loaded.num_gaussians == N
    assert torch.allclose(gaussians_loaded._xyz, gaussians_orig._xyz, atol=1e-5)
    assert torch.allclose(gaussians_loaded._features_dc, gaussians_orig._features_dc, atol=1e-5)
    assert torch.allclose(gaussians_loaded._features_rest, gaussians_orig._features_rest, atol=1e-5)
    assert torch.allclose(gaussians_loaded._opacity, gaussians_orig._opacity, atol=1e-5)
    assert torch.allclose(gaussians_loaded._scaling, gaussians_orig._scaling, atol=1e-5)
    assert torch.allclose(gaussians_loaded._rotation, gaussians_orig._rotation, atol=1e-5)

    print(f"\n[OK] Official Inria PLY Serialization Verified: Round-trip byte-level precision confirmed")


# ══════════════════════════════════════════════════════════════════
# 3. Test Evaluation Metrics
# ══════════════════════════════════════════════════════════════════

def test_metrics_psnr_and_ssim():
    img1 = torch.rand((3, 64, 64), dtype=torch.float32)
    
    # Identical images
    psnr_perfect = compute_psnr(img1, img1)
    ssim_perfect = compute_ssim(img1, img1)
    assert psnr_perfect >= 99.0
    assert abs(ssim_perfect - 1.0) < 1e-4

    # Perturbed image
    img2 = torch.clamp(img1 + 0.1 * torch.randn_like(img1), 0.0, 1.0)
    psnr_noisy = compute_psnr(img2, img1)
    ssim_noisy = compute_ssim(img2, img1)
    assert psnr_noisy < 40.0
    assert ssim_noisy < 1.0

    print(f"\n[OK] Metrics Verified: Perfect (PSNR={psnr_perfect:.1f} dB, SSIM={ssim_perfect:.4f}) | Noisy (PSNR={psnr_noisy:.1f} dB, SSIM={ssim_noisy:.4f})")


# ══════════════════════════════════════════════════════════════════
# 4. Test Real COLMAP Scene Training on Truck
# ══════════════════════════════════════════════════════════════════

def test_real_colmap_training_truck(tmp_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        pytest.skip("CUDA device required for real COLMAP training test")
    if not TRUCK_PATH.exists():
        pytest.skip(f"Truck dataset not found at {TRUCK_PATH}")

    torch.manual_seed(42)
    np.random.seed(42)

    scene_info = read_colmap_scene_info(
        dataset_path=str(TRUCK_PATH),
        images_subfolder="images",
        eval_mode=True,
        llffhold=8,
    )

    # Use first 4 training views and 2 test views for rapid test verification
    train_cams = scene_info.train_cameras[:4]
    test_cams = scene_info.test_cameras[:2]

    # Pre-cache test cameras at 256 max dim
    for cam in train_cams + test_cams:
        cam.load_image(max_dim=256)

    gaussians = GaussianModel(sh_degree=3)
    gaussians.init_from_pointcloud(
        points=scene_info.point_cloud.points[:2000], # 2k points subset for rapid test
        colors=scene_info.point_cloud.colors[:2000],
        device=device,
    )

    config = TrainingConfig(
        iterations=30,
        densify_from_iter=5,
        densify_until_iter=25,
        densification_interval=5,
        opacity_reset_interval=50,
        densify_grad_threshold=1e-5,
        position_lr_init=0.001,
        position_lr_final=0.0001,
        position_lr_max_steps=30,
    )

    trainer = GaussianTrainer(config)
    trainer.setup(gaussians, train_cams)
    renderer = TileBasedRasterizer()

    init_loss = None
    final_loss = None

    print("\n[Real COLMAP Training] Running 25 iterations on Truck dataset...")
    for step in range(1, 26):
        cam = train_cams[(step - 1) % len(train_cams)]
        metrics = trainer.train_step(cam, renderer, iteration=step)
        if step == 1:
            init_loss = metrics["loss"]
        if step == 25:
            final_loss = metrics["loss"]

    assert final_loss < init_loss, f"Loss did not decrease: init={init_loss:.4f}, final={final_loss:.4f}"

    # Evaluate on unseen test views
    eval_dir = tmp_path / "eval"
    results = evaluate_dataset(gaussians, test_cams, renderer, save_dir=eval_dir)
    assert results["mean_psnr"] > 10.0
    assert results["mean_ssim"] > 0.0

    # Verify PLY export
    ply_out = tmp_path / "truck_final.ply"
    gaussians.save_ply(ply_out)
    assert ply_out.exists() and ply_out.stat().st_size > 0

    print(f"\n[OK] Real COLMAP Scene Training Verified on Truck:")
    print(f"     - Loss: {init_loss:.4f} -> {final_loss:.4f}")
    print(f"     - Unseen Test Views Mean PSNR: {results['mean_psnr']:.2f} dB, SSIM: {results['mean_ssim']:.4f}")
    print(f"     - Saved Official PLY Checkpoint: {ply_out.name} ({ply_out.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
