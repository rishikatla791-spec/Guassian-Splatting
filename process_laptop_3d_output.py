#!/usr/bin/env python3
"""
process_laptop_3d_output.py — Export & Render 3D Model from 5 Uploaded Laptop Images
"""
import os
import json
import math
import sys
from pathlib import Path
import numpy as np

root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

from export_3d_mesh import export_all_formats, load_gaussian_ply
from render_3d_preview import render_camera_view

def main():
    output_dir = root_dir / "output_laptop_3dmodel"
    ply_file = output_dir / "point_cloud" / "iteration_500" / "point_cloud.ply"
    
    if not ply_file.exists():
        found_plys = list(output_dir.glob("**/point_cloud.ply"))
        if found_plys:
            ply_file = found_plys[-1]
            print(f"[Notice] Using available checkpoint: {ply_file}")
        else:
            print(f"[Error] No point_cloud.ply found in {output_dir}")
            return

    print("==========================================================================")
    print("  PROCESSING & EXPORTING 3D LAPTOP MODEL FROM 5 UPLOADED IMAGES")
    print("==========================================================================")
    
    # 1. Export 3D Mesh formats (OBJ, GLTF, PLY)
    mesh_paths = export_all_formats(ply_file, output_dir)
    print(f"[OK] Saved 3D Wavefront OBJ model: {mesh_paths['obj']}")
    print(f"[OK] Saved 3D Polygonal PLY mesh:   {mesh_paths['ply_mesh']}")
    print(f"[OK] Saved 3D Standard GLTF asset:  {mesh_paths['gltf']}")

    # 2. Load Gaussians & Render Multi-Angle Preview Images
    pts, colors, opacities, scales = load_gaussian_ply(ply_file)
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    render_paths = []
    
    for angle in angles:
        img = render_camera_view(pts, colors, opacities, scales, azimuth_deg=angle, dist=3.0)
        out_img_path = output_dir / f"laptop_preview_{angle}deg.png"
        img.save(out_img_path)
        render_paths.append(str(out_img_path.resolve()).replace("\\", "/"))
        print(f"[OK] Rendered Laptop Preview View {angle}° -> {out_img_path.name}")

    # 3. Update HTML viewer with laptop 3D scene data
    gaussians_js_data = []
    step = 1 if len(pts) <= 4000 else math.ceil(len(pts) / 3600)
    for i in range(0, len(pts), step):
        r, g, b = (colors[i] * 255).astype(int).tolist()
        px, py, pz = pts[i].tolist()
        s0, s1, s2 = scales[i].tolist()
        gaussians_js_data.append({
            "id": i,
            "pos": [round(px, 4), round(py, 4), round(pz, 4)],
            "normal": [0.0, 1.0, 0.0],
            "color": [r, g, b],
            "scales": [round(s0, 4), round(s1, 4), round(s2, 4)],
            "type": "reconstructed_laptop"
        })

    html_file = root_dir / "index.html"
    if html_file.exists():
        with open(html_file, "r", encoding="utf-8") as f:
            html_content = f.read()

        angles_data_js = "const anglesData = [\n"
        labels = ["0° (Front View)", "45° (Front-Right)", "90° (Right Side)", "135° (Back-Right)",
                  "180° (Direct Back)", "225° (Back-Left)", "270° (Left Side)", "315° (High Overhead)"]
        agents = ["View-01 (Front)", "View-02 (Front-Right)", "View-03 (Right Side)", "View-04 (Back-Right)",
                  "View-05 (Direct Back)", "View-06 (Back-Left)", "View-07 (Left Side)", "View-08 (Overhead)"]

        for i, angle in enumerate(angles):
            img_uri = f"file:///{render_paths[i]}"
            angles_data_js += f"""      {{
            angle: {angle},
            label: "{labels[i]}",
            img: "{img_uri}",
            flipped: false,
            agent: "{agents[i]}"
          }}{',' if i < len(angles)-1 else ''}\n"""
        angles_data_js += "    ];"

        import re
        html_content = re.sub(r'const anglesData = \[[\s\S]*?\];', angles_data_js, html_content)

        scene_replace_js = f"""  generateScene() {{
        this.gaussians = {json.dumps(gaussians_js_data)};
        document.getElementById('telemetry-count').innerText = this.gaussians.length.toLocaleString();
      }}"""
        html_content = re.sub(
            r'generateScene\(\) \{[\s\S]*?document\.getElementById\(\'telemetry-count\'\)\.innerText = this\.gaussians\.length\.toLocaleString\(\);\s*\}',
            scene_replace_js,
            html_content
        )

        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"[OK] Updated {html_file.name} viewer with 3D Laptop Splat model!")

if __name__ == "__main__":
    main()
