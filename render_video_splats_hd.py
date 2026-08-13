"""
HD Renderer for Video Apple 3D Gaussian Model.
"""
import sys
import shutil
import os
from pathlib import Path

import numpy as np
import torch
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from gaussian.core.gaussians import GaussianModel
from gaussian.core.camera import Camera, CameraIntrinsics, CameraExtrinsics
from gaussian.renderer.tile_rasterizer import TileBasedRasterizer


def render_hd_views():
    ply_path = Path(r"C:\Users\Rishi\Downloads\gaussian\output_apple_video_3dmodel\point_cloud\iteration_200\point_cloud.ply")
    if not ply_path.exists():
        print(f"PLY not found: {ply_path}")
        return

    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(str(ply_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gaussians = gaussians.to(device)

    rasterizer = TileBasedRasterizer()
    out_dir = Path(r"C:\Users\Rishi\Downloads\gaussian\imgaes")
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir = Path(r"C:\Users\Rishi\.gemini\antigravity-cli\brain\a0adbc4a-f67b-4f6c-9c96-a85dcf1a26eb")

    angles = [0, 45, 90, 180]
    w, h = 960, 540
    f_approx = 1.25 * max(w, h)
    cx, cy = w / 2.0, h / 2.0
    bg = torch.ones(3, device=device)  # white bg

    for deg in angles:
        rad = np.radians(deg)
        eye = np.array([2.8 * np.sin(rad), 0.2, 2.8 * np.cos(rad)], dtype=np.float64)
        target = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        intr = CameraIntrinsics(fx=f_approx, fy=f_approx, cx=cx, cy=cy, width=w, height=h)
        extr = CameraExtrinsics.from_look_at(eye, target, up)
        cam = Camera(uid=deg, intrinsics=intr, extrinsics=extr).to(device)

        with torch.no_grad():
            out = rasterizer.render(gaussians, cam, bg_color=bg)
            render_img = (out["render"].clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            render_bgr = cv2.cvtColor(render_img, cv2.COLOR_RGB2BGR)

        out_name = f"video_apple_splats_{deg}deg.png"
        out_file = out_dir / out_name
        cv2.imwrite(str(out_file), render_bgr)
        shutil.copy(str(out_file), str(art_dir / out_name))
        print(f"Saved & copied: {out_name}")

    print("[OK] Video apple HD rendering complete.")


if __name__ == "__main__":
    render_hd_views()
