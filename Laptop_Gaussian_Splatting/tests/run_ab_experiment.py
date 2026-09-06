"""
Phase 4: Short A/B Experiment on Real Truck COLMAP Scene.

Compares:
  Model A: Previous implementation (Raw pixel gradient accumulation)
  Model B: Verified NDC-correct implementation (Scaled by 0.5*W, 0.5*H matching Inria backward.cu)

Fixed Settings (Identical across A and B):
  - Dataset: imgaes/truck (251 views, 219 train, 32 test)
  - Initial Point Cloud: 136,029 points
  - Resolution: max_dim = 512
  - Random Seed: 42
  - Iterations: 700 steps (densification active at step 500, 600, 700)
  - Densification threshold: 0.0002 (official default)
  - Loss formulation: 0.8 * L1 + 0.2 * (1 - SSIM)
"""

import sys
import time
from pathlib import Path
import torch
import numpy as np

root_dir = Path(__file__).parent.parent.resolve()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from scene.dataset_readers import read_colmap_scene_info
from core.gaussians import GaussianModel
from renderer.tile_rasterizer import TileBasedRasterizer
from training.config import TrainingConfig
from training.trainer import GaussianTrainer
from training.metrics import evaluate_dataset


def run_experiment(mode: str, seed: int = 42, steps: int = 700):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda"

    source_path = root_dir / "imgaes" / "truck"
    scene_info = read_colmap_scene_info(
        dataset_path=str(source_path),
        images_subfolder="images",
        eval_mode=True,
        llffhold=8,
    )

    train_cams = scene_info.train_cameras
    test_cams = scene_info.test_cameras

    for cam in train_cams:
        cam.load_image(max_dim=512)
    for cam in test_cams:
        cam.load_image(max_dim=512)

    gaussians = GaussianModel(sh_degree=0)
    gaussians.init_from_pointcloud(
        points=scene_info.point_cloud.points,
        colors=scene_info.point_cloud.colors,
        device=device,
    )

    config = TrainingConfig(
        iterations=steps,
        densify_grad_threshold=0.0002,
        densify_from_iter=500,
        densify_until_iter=15_000,
        densification_interval=100,
        opacity_reset_interval=3_000,
        white_background=False,
    )

    trainer = GaussianTrainer(config)
    trainer.setup(gaussians, train_cams)
    renderer = TileBasedRasterizer()

    total_clones = 0
    total_splits = 0
    total_pruned = 0
    grad_norms = []

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        cam_idx = np.random.randint(0, len(train_cams))
        cam = train_cams[cam_idx]

        # Forward
        render_out = renderer.render(gaussians, cam, bg_color=torch.zeros(3, device=device))
        rendered = render_out["render"]
        viewspace_points = render_out["viewspace_points"]
        visibility_filter = render_out["visibility_filter"]
        radii = render_out["radii"]

        gt_image = cam.load_image().to(device)
        if gt_image.shape[-2:] != rendered.shape[-2:]:
            import torch.nn.functional as F
            gt_image = F.interpolate(gt_image.unsqueeze(0), size=rendered.shape[-2:], mode='bilinear', align_corners=False).squeeze(0)

        from training.loss import combined_loss, l1_loss
        loss = combined_loss(rendered, gt_image, lambda_dssim=0.2)
        loss_val = loss.item()

        trainer.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Densification accumulation
        if step < config.densify_until_iter:
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter],
                radii[visibility_filter].float()
            )
            if mode == "A_raw_pixel":
                # Model A: Raw pixel gradient (no W/2, H/2 scaling)
                gaussians.add_densification_stats(viewspace_points, visibility_filter)
            else:
                # Model B: NDC-scaled gradient (matching official Inria)
                gaussians.add_densification_stats(
                    viewspace_points,
                    visibility_filter,
                    image_width=cam.width,
                    image_height=cam.height,
                )

            # Record average gradient magnitude before reset
            if step in (500, 600, 700):
                avg_grad = (gaussians.xyz_gradient_accum / (gaussians.denom + 1e-8)).mean().item()
                grad_norms.append(avg_grad)

            # Densify / Prune
            if step > config.densify_from_iter and step % config.densification_interval == 0:
                scene_extent = trainer._estimate_scene_extent()
                # Track clone/split
                n_init = gaussians.num_gaussians
                grads = gaussians.xyz_gradient_accum / (gaussians.denom + 1e-8)
                grads[grads.isnan()] = 0.0

                n_clone = gaussians.densify_and_clone(grads, config.densify_grad_threshold, scene_extent, trainer.optimizer)
                n_split = gaussians.densify_and_split(grads, config.densify_grad_threshold, scene_extent, trainer.optimizer)
                
                prune_mask = (gaussians.get_opacity < config.min_opacity).squeeze()
                n_prune = int(prune_mask.sum().item())
                gaussians.prune_points(prune_mask, trainer.optimizer)

                gaussians.xyz_gradient_accum.zero_()
                gaussians.denom.zero_()
                gaussians.max_radii2D.zero_()

                total_clones += n_clone
                total_splits += n_split
                total_pruned += n_prune

        trainer.optimizer.step()
        trainer._update_lr(step)

    torch.cuda.synchronize()
    total_time = time.perf_counter() - t0
    peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # Evaluate on test set
    eval_res = evaluate_dataset(
        gaussians=gaussians,
        test_cameras=test_cams,
        renderer=renderer,
        compute_lpips_metric=False,
    )

    return {
        "mode": mode,
        "mean_grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
        "clones": total_clones,
        "splits": total_splits,
        "pruned": total_pruned,
        "final_gaussians": gaussians.num_gaussians,
        "final_loss": loss_val,
        "test_psnr": eval_res["mean_psnr"],
        "test_ssim": eval_res["mean_ssim"],
        "peak_vram_mb": peak_alloc_mb,
        "duration_sec": total_time,
    }


def main():
    print("=" * 90)
    print("  RUNNING SHORT A/B EXPERIMENT (700 STEPS ON REAL TRUCK DATASET)")
    print("=" * 90)

    print("\n>>> Running Model A (Current / Raw Pixel Gradients)...")
    res_a = run_experiment(mode="A_raw_pixel", seed=42, steps=700)

    print("\n>>> Running Model B (Verified NDC-Correct Gradients)...")
    res_b = run_experiment(mode="B_ndc_scaled", seed=42, steps=700)

    print("\n" + "=" * 90)
    print("  A/B EXPERIMENTAL RESULTS COMPARISON TABLE")
    print("=" * 90)
    print(f"{'Metric':<30} | {'Model A (Raw Pixel)':<25} | {'Model B (NDC Scaled)':<25}")
    print("-" * 90)
    print(f"{'Avg Gradient Magnitude':<30} | {res_a['mean_grad_norm']:<25.6e} | {res_b['mean_grad_norm']:<25.6e}")
    print(f"{'Densification Threshold':<30} | {'0.000200':<25} | {'0.000200':<25}")
    print(f"{'Gaussians Cloned':<30} | {res_a['clones']:<25,} | {res_b['clones']:<25,}")
    print(f"{'Gaussians Split':<30} | {res_a['splits']:<25,} | {res_b['splits']:<25,}")
    print(f"{'Gaussians Pruned':<30} | {res_a['pruned']:<25,} | {res_b['pruned']:<25,}")
    print(f"{'Final Gaussian Count':<30} | {res_a['final_gaussians']:<25,} | {res_b['final_gaussians']:<25,}")
    print(f"{'Final Loss':<30} | {res_a['final_loss']:<25.4f} | {res_b['final_loss']:<25.4f}")
    print(f"{'Test Mean PSNR':<30} | {res_a['test_psnr']:<25.2f} dB | {res_b['test_psnr']:<25.2f} dB")
    print(f"{'Test Mean SSIM':<30} | {res_a['test_ssim']:<25.4f} | {res_b['test_ssim']:<25.4f}")
    print(f"{'Peak VRAM':<30} | {res_a['peak_vram_mb']:<25.1f} MB | {res_b['peak_vram_mb']:<25.1f} MB")
    print(f"{'Runtime Duration':<30} | {res_a['duration_sec']:<25.1f} s | {res_b['duration_sec']:<25.1f} s")
    print("=" * 90)


if __name__ == "__main__":
    main()
