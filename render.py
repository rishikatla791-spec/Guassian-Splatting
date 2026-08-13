#!/usr/bin/env python3
"""
render.py — Render novel views from a trained 3DGS model.

Usage:
    # Render test cameras from COLMAP scene
    python render.py \\
        --model_path ./output \\
        --source_path /path/to/colmap/sparse/0 \\
        --images_path /path/to/images \\
        --output_dir ./renders

    # Render 360° orbit video
    python render.py \\
        --model_path ./output \\
        --orbit \\
        --orbit_frames 240 \\
        --output_video orbit.mp4

    # Launch interactive viewer
    python render.py --model_path ./output --viewer
"""

import argparse
import sys
from pathlib import Path
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render from trained 3DGS model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", "-m", type=str, required=True,
                        help="Path to model output directory")
    parser.add_argument("--iteration", type=int, default=-1,
                        help="Specific iteration to load (-1 = latest)")
    parser.add_argument("--source_path", type=str, default=None)
    parser.add_argument("--images_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Save rendered images here")
    parser.add_argument("--orbit", action="store_true",
                        help="Render 360° orbit video")
    parser.add_argument("--orbit_frames", type=int, default=120)
    parser.add_argument("--orbit_radius", type=float, default=3.0)
    parser.add_argument("--output_video", type=str, default="orbit.mp4")
    parser.add_argument("--viewer", action="store_true",
                        help="Launch interactive viewer")
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--white_background", action="store_true")
    return parser.parse_args()


def find_latest_ply(model_path: str) -> str:
    """Find the latest saved point_cloud.ply file."""
    model_path = Path(model_path)
    pc_dir = model_path / "point_cloud"
    if not pc_dir.exists():
        raise FileNotFoundError(f"No point_cloud directory in {model_path}")

    iterations = sorted(
        [int(d.name.split('_')[1]) for d in pc_dir.iterdir() if d.is_dir()],
        reverse=True
    )
    if not iterations:
        raise FileNotFoundError("No saved iterations found")
    latest = iterations[0]
    ply = pc_dir / f"iteration_{latest}" / "point_cloud.ply"
    print(f"Loading iteration {latest}: {ply}")
    return str(ply)


def main():
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from gaussian.core.gaussians import GaussianModel
    from gaussian.ui.viewer import GaussianViewer, render_360_video

    # Load model
    g = GaussianModel(sh_degree=args.sh_degree)
    ply_path = (
        find_latest_ply(args.model_path) if args.iteration == -1
        else str(Path(args.model_path) / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply")
    )
    g.load_ply(ply_path)

    for p in g.parameters():
        p.data = p.data.to(args.device)
    g.xyz_gradient_accum = g.xyz_gradient_accum.to(args.device)
    g.denom = g.denom.to(args.device)
    g.max_radii2D = g.max_radii2D.to(args.device)

    print(f"Loaded: {g}")

    if args.viewer:
        viewer = GaussianViewer(g)
        viewer.interactive_loop()

    elif args.orbit:
        render_360_video(
            g,
            output_path=args.output_video,
            n_frames=args.orbit_frames,
            radius=args.orbit_radius,
        )

    elif args.source_path and args.output_dir:
        from gaussian.pipeline import Pipeline
        config = {
            "source_path": args.source_path,
            "images_path": args.images_path or str(Path(args.source_path).parent.parent / "images"),
            "model_path": args.model_path,
            "sh_degree": args.sh_degree,
            "white_background": args.white_background,
        }
        pipeline = Pipeline(config)
        scene = pipeline.load_scene(
            colmap_path=args.source_path,
            images_path=config["images_path"],
        )
        all_cams = scene.train_cameras + scene.test_cameras
        pipeline.render_views(g, all_cams, args.output_dir, device=args.device)

    else:
        print("No action specified. Use --viewer, --orbit, or provide --source_path + --output_dir")
        print("Launch interactive viewer with: python render.py --model_path ./output --viewer")


if __name__ == "__main__":
    main()
