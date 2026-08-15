#!/usr/bin/env python3
"""
render_novel_views.py — Render trained 3DGS model from novel viewpoints.
"""
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from core.gaussians import GaussianModel
from core.camera import Camera, CameraIntrinsics, CameraExtrinsics
from renderer.tile_rasterizer import TileBasedRasterizer

def main():
    ply_path = root_dir / "output_white_laptop_3dmodel" / "point_cloud" / "iteration_300" / "point_cloud.ply"
    out_dir = root_dir / "output_white_laptop_3dmodel" / "novel_views"
    out_dir.mkdir(parents=True, exist_ok=True)

    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(str(ply_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gaussians = gaussians.to(device)

    rasterizer = TileBasedRasterizer()
    bg_color = torch.zeros(3, device=device)

    # Intrinsics: 512x512
    intrinsics = CameraIntrinsics(fx=400.0, fy=400.0, cx=256.0, cy=256.0, width=512, height=512)

    # 4 Orbit viewpoints around laptop
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    radius = 2.8

    print(f"=== Rendering 8 Novel Orbit Views of Reconstructed White Laptop ===")
    with torch.no_grad():
        for idx, deg in enumerate(angles):
            rad = np.radians(deg)
            eye = np.array([radius * np.sin(rad), 0.8, radius * np.cos(rad)], dtype=np.float32)
            target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

            extrinsics = CameraExtrinsics.from_look_at(eye, target, up)
            cam = Camera(uid=idx, intrinsics=intrinsics, extrinsics=extrinsics).to(device)

            out = rasterizer.render(gaussians, cam, bg_color=bg_color)
            render_img = out["render"].clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
            img_uint8 = (render_img * 255.0).astype(np.uint8)

            save_path = out_dir / f"novel_view_deg_{deg:03d}.png"
            Image.fromarray(img_uint8).save(save_path)
            print(f"  [OK] Saved novel orbit render: {save_path.name}")

    print(f"[OK] Novel view rendering complete! Assets saved in: {out_dir}")

if __name__ == "__main__":
    main()
