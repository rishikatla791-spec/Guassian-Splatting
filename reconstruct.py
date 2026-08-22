#!/usr/bin/env python3
"""
reconstruct.py — Unified 3D Reconstruction Pipeline Entrypoint

Usage:
    python reconstruct.py --images_path ./imgaes --output_dir ./output_reconstruction
    python reconstruct.py --images_path /path/to/photos --iterations 30000 --backend auto
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from pipeline import ReconstructionPipeline
except ImportError:
    from gaussian.pipeline import ReconstructionPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="High-Fidelity Multi-View Image-to-3D Reconstruction Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--images_path", "-i", type=str, default="./imgaes",
                        help="Directory containing input multi-view images")
    parser.add_argument("--output_dir", "-o", type=str, default="./output_reconstruction",
                        help="Directory to save 3D reconstruction checkpoints and PLY model")
    parser.add_argument("--iterations", type=int, default=15000,
                        help="Total 3DGS training & refinement iterations")
    parser.add_argument("--sh_degree", type=int, default=3,
                        help="Spherical Harmonics degree (0 to 3)")
    parser.add_argument("--backend", type=str, default="auto",
                        choices=["auto", "colmap", "dust3r", "feature_matching"],
                        help="Pose estimation backend strategy")
    return parser.parse_args()


def main():
    args = parse_args()

    config = {
        "images_path": args.images_path,
        "output_dir": args.output_dir,
        "iterations": args.iterations,
        "sh_degree": args.sh_degree,
        "backend": args.backend,
    }

    print("==========================================================================")
    print("  3D GAUSSIAN SPLATTING MULTI-VIEW RECONSTRUCTION ENGINE")
    print(f"  Config: Images='{args.images_path}' | Output='{args.output_dir}' | Iterations={args.iterations}")
    print("==========================================================================\n")

    pipeline = ReconstructionPipeline(config)
    gaussians, metrics = pipeline.run_full_pipeline()

    print("\n[OK] Reconstruction pipeline execution completed successfully!")


if __name__ == "__main__":
    main()
