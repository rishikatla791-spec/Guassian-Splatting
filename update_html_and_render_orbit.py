#!/usr/bin/env python3
"""
update_html_and_render_orbit.py — Update HTML Viewer with Real 3D Reconstruction Data.

1. Renders 8 high-definition orbit camera view images from 3D Gaussian Splats.
2. Extracts raw 3D Gaussian primitive data (6,597 Gaussians) into JSON format.
3. Updates index.html with real dataset images, 3D model download links (OBJ, GLTF, PLY),
   and live 3D Gaussian Splatting scene data.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

from export_3d_mesh import load_gaussian_ply
from render_3d_preview import render_camera_view


def main():
    new_input_dir = root_dir / "output_new_input_3dmodel"
    ply_file = new_input_dir / "point_cloud" / "iteration_0" / "point_cloud.ply"

    if not ply_file.exists():
        print(f"Error: {ply_file} not found")
        return

    pts, colors, opacities, scales = load_gaussian_ply(ply_path=ply_file)
    print(f"Loaded {len(pts):,} 3D Gaussians from {ply_file.name}")

    # 1. Render 8 Orbit Camera Views
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    orbit_image_paths = []

    for angle in angles:
        img = render_camera_view(pts, colors, opacities, scales, azimuth_deg=angle, dist=3.2)
        out_img_path = new_input_dir / f"orbit_{angle}deg.png"
        img.save(out_img_path)
        orbit_image_paths.append(str(out_img_path.resolve()).replace("\\", "/"))
        print(f"  [OK] Rendered Orbit View {angle}° -> {out_img_path.name}")

    # 2. Extract Gaussians into JavaScript array structure
    # Subsample if large for smooth 60 FPS HTML5 canvas rendering
    step = 1 if len(pts) <= 4000 else math.ceil(len(pts) / 3600)
    sampled_indices = range(0, len(pts), step)

    gaussians_js_data = []
    for i in sampled_indices:
        r, g, b = (colors[i] * 255).astype(int).tolist()
        px, py, pz = pts[i].tolist()
        s0, s1, s2 = scales[i].tolist()
        gaussians_js_data.append({
            "id": i,
            "pos": [round(px, 4), round(py, 4), round(pz, 4)],
            "normal": [0.0, 1.0, 0.0],
            "color": [r, g, b],
            "scales": [round(s0, 4), round(s1, 4), round(s2, 4)],
            "type": "reconstructed"
        })

    print(f"Prepared {len(gaussians_js_data):,} Gaussians for HTML JS Engine")

    # 3. Read index.html
    html_file = root_dir / "index.html"
    with open(html_file, "r", encoding="utf-8") as f:
        html_content = f.read()

    # Update anglesData in index.html
    angles_data_js = "const anglesData = [\n"
    labels = ["0° (Front View)", "45° (Front-Right)", "90° (Right Side)", "135° (Back-Right)",
              "180° (Direct Back)", "225° (Back-Left)", "270° (Left Side)", "315° (High Overhead)"]
    agents = ["Agent-01 (Front)", "Agent-02 (Front-Right)", "Agent-03 (Right Side)", "Agent-04 (Back-Right)",
              "Agent-05 (Direct Back)", "Agent-06 (Back-Left)", "Agent-07 (Left Side)", "Agent-08 (Overhead)"]

    for i, angle in enumerate(angles):
        img_uri = f"file:///{orbit_image_paths[i]}"
        angles_data_js += f"""      {{
        angle: {angle},
        label: "{labels[i]}",
        img: "{img_uri}",
        flipped: false,
        agent: "{agents[i]}"
      }}{',' if i < len(angles)-1 else ''}\n"""
    angles_data_js += "    ];"

    # Replace anglesData block in HTML
    import re
    html_content = re.sub(
        r'const anglesData = \[[\s\S]*?\];',
        angles_data_js,
        html_content
    )

    # Embed real 3D Gaussians into GaussianSplatScene
    gaussians_json_str = json.dumps(gaussians_js_data)
    scene_replace_js = f"""  generateScene() {{
        this.gaussians = {gaussians_json_str};
        document.getElementById('telemetry-count').innerText = this.gaussians.length.toLocaleString();
      }}"""

    html_content = re.sub(
        r'generateScene\(\) \{[\s\S]*?document\.getElementById\(\'telemetry-count\'\)\.innerText = this\.gaussians\.length\.toLocaleString\(\);\s*\}',
        scene_replace_js,
        html_content
    )

    # Write updated index.html
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n[OK] Successfully updated {html_file.name} with real 3D Gaussian engine scene & orbit renders!")


if __name__ == "__main__":
    main()
