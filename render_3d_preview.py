#!/usr/bin/env python3
"""
render_3d_preview.py — 3D Model & Splat Multi-Angle High-Resolution Offscreen Renderer.

Loads 3D Gaussian Splats / PLY / OBJ models and renders high-definition multi-view
perspective images (0°, 90°, 180°, 270°) for visual presentation.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

file_path = Path(__file__).resolve()
sys.path.insert(0, str(file_path.parent.parent))
sys.path.insert(0, str(file_path.parent))

from export_3d_mesh import load_gaussian_ply


def render_camera_view(
    pts: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
    scales: np.ndarray,
    azimuth_deg: float,
    elevation_deg: float = 15.0,
    dist: float = 3.2,
    img_w: int = 1024,
    img_h: int = 1024,
) -> Image.Image:
    """
    Render 3D point cloud / splats into a high-definition image.
    """
    img = Image.new("RGB", (img_w, img_h), (14, 19, 32))  # Studio dark backdrop
    draw = ImageDraw.Draw(img)

    # Convert angles to radians
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)

    cam_x = dist * math.cos(el) * math.sin(az)
    cam_y = dist * math.sin(el) + 0.1
    cam_z = dist * math.cos(el) * math.cos(az)

    cam_pos = np.array([cam_x, cam_y, cam_z], dtype=np.float32)
    target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    z_axis = cam_pos - target
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    R_cam = np.vstack([x_axis, y_axis, z_axis])  # (3, 3)

    # Center points
    p_center = np.mean(pts, axis=0)
    pts_centered = pts - p_center

    # Transform to camera space
    pts_cam = (R_cam @ pts_centered.T).T  # (N, 3)

    # Sort by depth (back to front for transparency blending)
    depths = pts_cam[:, 2]
    sort_idx = np.argsort(depths)[::-1]

    # Perspective projection
    f_len = 1.3 * max(img_w, img_h)
    cx, cy = img_w / 2.0, img_h / 2.0

    # Render grid shadow plane
    shadow_y = cy + (0.8 * f_len / dist)
    draw.ellipse([cx - 220, shadow_y - 45, cx + 220, shadow_y + 45], fill=(8, 11, 18))

    for idx in sort_idx:
        z = pts_cam[idx, 2]
        if z <= 0.1:
            continue

        x_proj = (pts_cam[idx, 0] * f_len / z) + cx
        y_proj = (-pts_cam[idx, 1] * f_len / z) + cy

        if 0 <= x_proj < img_w and 0 <= y_proj < img_h:
            sc = np.mean(scales[idx])
            radius = max(2.5, min(35.0, (sc * f_len / z) * 1.8))
            opac = opacities[idx]
            col = (colors[idx] * 255.0).astype(np.uint8)
            col_tuple = (int(col[0]), int(col[1]), int(col[2]))

            draw.ellipse(
                [x_proj - radius, y_proj - radius, x_proj + radius, y_proj + radius],
                fill=col_tuple
            )

    return img


def render_all_views(ply_file: str | Path, out_dir: str | Path) -> list[str]:
    ply_file = Path(ply_file)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pts, colors, opacities, scales = load_gaussian_ply(ply_file)
    print(f"=== Rendering 3D Preview Images for {ply_file.name} ({len(pts):,} points) ===")

    angles = [0, 90, 180, 270]
    out_paths = []

    for angle in angles:
        img = render_camera_view(pts, colors, opacities, scales, azimuth_deg=angle)
        img_filename = out_dir / f"render_{angle}deg.png"
        img.save(img_filename)
        out_paths.append(str(img_filename))
        print(f"  [OK] Saved {angle}° View Render -> {img_filename.name}")

    return out_paths


def main():
    # Render new input model views
    new_input_ply = Path("./output_new_input_3dmodel/point_cloud/iteration_0/point_cloud.ply")
    new_input_out = Path("./output_new_input_3dmodel")
    if new_input_ply.exists():
        render_all_views(new_input_ply, new_input_out)

    # Render apple video model views
    apple_ply = Path("./output_apple_video_3dmodel/point_cloud/iteration_500/point_cloud.ply")
    if not apple_ply.exists():
        apple_ply = Path("./output_apple_video_3dmodel/point_cloud/iteration_200/point_cloud.ply")
    apple_out = Path("./output_apple_video_3dmodel")
    if apple_ply.exists():
        render_all_views(apple_ply, apple_out)


if __name__ == "__main__":
    main()
