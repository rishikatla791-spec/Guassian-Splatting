#!/usr/bin/env python3
"""
train.py — Main training entry point for 3D Gaussian Splatting.

Usage:
    python train.py \\
        --source_path /path/to/colmap/sparse/0 \\
        --images_path /path/to/images \\
        --model_path  ./output \\
        --iterations  30000 \\
        --sh_degree   3

Or use the Pipeline API directly:
    from gaussian.pipeline import Pipeline
    pipeline = Pipeline(config={...})
    scene = pipeline.load_scene()
    gaussians = pipeline.train(scene)
"""

import argparse
import sys
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a 3D Gaussian Splatting scene",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source_path", "-s", type=str, required=True,
                        help="Path to COLMAP sparse reconstruction (cameras.bin + points3D.bin)")
    parser.add_argument("--images_path", "-i", type=str, default=None,
                        help="Path to images directory (default: source_path/../images)")
    parser.add_argument("--model_path", "-m", type=str, default="./output",
                        help="Output directory for trained model")
    parser.add_argument("--iterations", type=int, default=30_000,
                        help="Total training iterations")
    parser.add_argument("--sh_degree", type=int, default=3, choices=[0, 1, 2, 3],
                        help="Spherical harmonics degree for view-dependent color")
    parser.add_argument("--white_background", action="store_true",
                        help="Use white background (default: black)")
    parser.add_argument("--resolution", type=int, default=-1,
                        help="Resize images to this size (-1 = original)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Training device: 'cuda' or 'cpu'")
    parser.add_argument("--test_fraction", type=float, default=0.1,
                        help="Fraction of images used for test/evaluation")
    parser.add_argument("--render_after", action="store_true",
                        help="Render test views after training")
    return parser.parse_args()


def main():
    args = parse_args()

    # Add parent to path so 'gaussian' package is importable
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from gaussian.pipeline import Pipeline

    images = args.images_path or str(Path(args.source_path).parent.parent / "images")

    config = {
        "source_path":      args.source_path,
        "images_path":      images,
        "model_path":       args.model_path,
        "sh_degree":        args.sh_degree,
        "white_background": args.white_background,
        "resolution":       args.resolution,
    }

    pipeline = Pipeline(config)
    scene    = pipeline.load_scene(
        colmap_path=args.source_path,
        images_path=images,
        test_fraction=args.test_fraction,
    )

    print(f"\nScene loaded: {scene}")
    print(f"Training for {args.iterations} iterations on {args.device}...\n")

    gaussians = pipeline.train(
        scene,
        output_dir=args.model_path,
        iterations=args.iterations,
        device=args.device,
    )

    # Evaluation
    if scene.test_cameras:
        print("\nEvaluating on test set...")
        metrics = pipeline.evaluate(gaussians, scene.test_cameras, device=args.device)
        print(f"Test PSNR:  {metrics['psnr']:.2f} dB")
        print(f"Test SSIM:  {metrics['ssim']:.4f}")
        print(f"N views:    {metrics['n_views']}")

    # Optional: render test views
    if args.render_after and scene.test_cameras:
        render_dir = Path(args.model_path) / "test_renders"
        print(f"\nRendering test views to {render_dir}...")
        pipeline.render_views(gaussians, scene.test_cameras, str(render_dir), device=args.device)

    print("\nTraining complete!")
    print(f"Model saved to: {args.model_path}")


if __name__ == "__main__":
    main()
