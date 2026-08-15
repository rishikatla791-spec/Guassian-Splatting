#!/usr/bin/env python3
"""
verify_gpu_reconstruction.py — Execute GPU-accelerated 3DGS training & benchmark performance.
"""
import sys
import time
import torch
import numpy as np
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from pipeline.reconstruction_pipeline import ReconstructionPipeline
from export_3d_mesh import export_all_formats
from core.gaussians import GaussianModel
from renderer.tile_rasterizer import TileBasedRasterizer
from training.trainer import GaussianTrainer
from training.config import TrainingConfig

def main():
    print("==========================================================================")
    print("   REAL GPU-ACCELERATED RECONSTRUCTION BENCHMARK & AUDIT                 ")
    print("==========================================================================\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing on Device: {device.upper()}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Model:           {gpu_name}")
        print(f"Total VRAM:          {vram_gb:.2f} GB")
        torch.cuda.reset_peak_memory_stats()

    images_dir = str(root_dir / "imgaes_white_laptop_validated")
    output_dir = str(root_dir / "output_white_laptop_3dmodel")

    cfg = {
        "images_path": images_dir,
        "output_dir": output_dir,
        "iterations": 120,
    }

    t0 = time.time()
    pipeline = ReconstructionPipeline(cfg)

    # 1. Pose Estimation
    cameras, raw_points3d, raw_colors3d, pose_metrics = pipeline.pose_estimator.estimate_poses(images_dir=pipeline.images_path)
    
    # 2. Object Mask Isolation
    cameras = pipeline.mask_generator.process_camera_list(cameras)

    # 3. Dense Geometry Reconstruction with Surface Normal Estimation
    clean_pts, clean_cols, normals, knn_dist = pipeline.geometry_reconstructor.process_point_cloud(
        points=raw_points3d,
        colors=raw_colors3d,
        cameras=cameras
    )

    # 4. Train / Test Split
    n_test = max(1, int(len(cameras) * 0.15))
    test_indices = set(range(0, len(cameras), max(1, len(cameras) // n_test)))
    train_cameras = [c for i, c in enumerate(cameras) if i not in test_indices]
    test_cameras  = [c for i, c in enumerate(cameras) if i in test_indices]

    # Pre-load camera buffers
    for c in cameras:
        c.load_image(max_dim=256)

    # 5. Gaussian Model Initialization
    gaussians = GaussianModel(sh_degree=pipeline.sh_degree)
    gaussians.init_from_pointcloud(clean_pts, clean_cols)
    gaussians = gaussians.to(device)

    # 6. Training & Densification
    train_cfg = TrainingConfig(
        iterations=120,
        model_path=str(pipeline.output_dir),
        sh_degree=pipeline.sh_degree,
        densify_from_iter=30,
        densify_until_iter=100,
        densification_interval=20,
        min_opacity=0.005,
    )
    renderer = TileBasedRasterizer()
    trainer = GaussianTrainer(train_cfg)

    train_cams = [c.to(device) for c in train_cameras]
    test_cams  = [c.to(device) for c in test_cameras]
    trainer.setup(gaussians, train_cams, test_cams)

    t_train_start = time.time()
    trained_gaussians = trainer.train(renderer)
    t_train_elapsed = time.time() - t_train_start

    # 7. Validation Evaluation
    val_metrics = pipeline.validation_suite.evaluate(
        gaussians=trained_gaussians,
        renderer=renderer,
        test_cameras=test_cams,
        points3d_gt=clean_pts,
    )

    t_total = time.time() - t0

    # VRAM Measurement
    vram_alloc_mb = 0.0
    vram_peak_mb = 0.0
    if device == "cuda":
        vram_alloc_mb = torch.cuda.memory_allocated(0) / (1024**2)
        vram_peak_mb  = torch.cuda.max_memory_allocated(0) / (1024**2)

    print("\n==========================================================================")
    print("   GPU-ACCELERATED RECONSTRUCTION RESULTS & BENCHMARK                    ")
    print("==========================================================================")
    print(f"• Execution Device:            {device.upper()}")
    if device == "cuda":
        print(f"• GPU Model:                  {gpu_name}")
        print(f"• Peak VRAM Allocated:        {vram_peak_mb:.2f} MB / {vram_gb*1024:.0f} MB")
        print(f"• Current VRAM In Use:        {vram_alloc_mb:.2f} MB")
    print(f"• 120-Step Training Time:      {t_train_elapsed:.2f}s ({120.0/t_train_elapsed:.2f} iters/sec)")
    print(f"• Total Pipeline Time:         {t_total:.2f}s")
    print(f"• Final Gaussian Count:        {trained_gaussians.num_gaussians:,}")
    print(f"• Holdout Test PSNR:           {val_metrics['psnr']:.2f} dB")
    print(f"• Holdout Test SSIM:           {val_metrics['ssim']:.4f}")
    print("==========================================================================\n")

if __name__ == "__main__":
    main()
