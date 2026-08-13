#!/usr/bin/env python3
"""
hulk_smash_multiview.py — Multi-Image Photorealistic 3D Gaussian Reconstruction Engine.

HULK SMASH MULTI-VIEW MODE:
Takes a bunch of images (multi-view photos from different angles around an object),
executes high-fidelity pose estimation, dense geometry synthesis, and PyTorch
3D Gaussian Splatting optimization, exporting an ultra-photorealistic 3D model!
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir.parent))

from gaussian.pipeline.reconstruction_pipeline import ReconstructionPipeline
from gaussian.pipeline.single_image_triposr import SingleImageTripoSRPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="HULK SMASH Multi-View Photorealistic 3D Reconstruction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--images_dir", "-i", type=str, default="./imgaes_apple_video",
                        help="Directory containing a bunch of multi-view photos")
    parser.add_argument("--output_dir", "-o", type=str, default="./output_new_input_3dmodel",
                        help="Output directory for photorealistic 3D PLY/OBJ models")
    parser.add_argument("--iterations", type=int, default=300,
                        help="Refinement iterations for ultra-photorealistic Splatting")
    return parser.parse_args()


def main():
    args = parse_args()
    img_dir = Path(args.images_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n==========================================================================")
    print("  [HULK SMASH] MULTI-IMAGE PHOTOREALISTIC 3D RECONSTRUCTION ENGINE")
    print(f"  Target Directory: '{img_dir}' | Iterations: {args.iterations}")
    print("==========================================================================\n")

    if not img_dir.exists():
        print(f"Error: Input images directory not found: {img_dir}")
        return

    # Find images
    image_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg")))
    print(f"[HULK SMASH] Loaded a bunch of {len(image_files)} multi-view images!")

    if len(image_files) == 0:
        print(f"Error: No image files found in {img_dir}")
        return

    # Run 9-Stage High-Fidelity Reconstruction
    config = {
        "images_path": str(img_dir),
        "output_dir": str(out_dir),
        "iterations": args.iterations,
        "sh_degree": 3,
        "backend": "auto"
    }

    t0 = time.time()
    recon_pipeline = ReconstructionPipeline(config)
    gaussians, metrics = recon_pipeline.run_full_pipeline()

    # Export PLY file to point_cloud/iteration_300/point_cloud.ply
    iter_dir = out_dir / f"point_cloud/iteration_{args.iterations}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    ply_path = iter_dir / "point_cloud.ply"
    gaussians.save_ply(str(ply_path))

    elapsed = time.time() - t0
    print(f"\n[HULK SMASH SUCCESS] Multi-View 3D Model Created in {elapsed:.2f}s!")
    print(f"[Saved] 3D Gaussian Model Saved: {ply_path}")

    # Automatically upgrade index.html viewer with new 3D model!
    try:
        import update_index_accurate_viewer
        update_index_accurate_viewer.main()
        print("\n🌐 Web Studio (index.html) successfully updated with your new 3D Model!")
    except Exception as e:
        print("Note: Run 'python update_index_accurate_viewer.py' to update index.html")


if __name__ == "__main__":
    main()
