#!/usr/bin/env python3
"""
run_single_image_laptop_3d.py — Generate high-density 3D model from front-view laptop photo
"""
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir.parent))

from gaussian.pipeline.single_image_triposr import SingleImageTripoSRPipeline
from gaussian.export_3d_mesh import export_all_formats

def main():
    img_path = root_dir / "imgaes_laptop_5views" / "view_01_WhatsApp Image 2026-08-14 at 01.05.22.jpeg"
    out_dir = root_dir / "output_laptop_single_image_3d"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("==========================================================================")
    print("  RUNNING SINGLE-IMAGE VOLUMETRIC 3D GENERATION FOR LAPTOP PHOTO")
    print("==========================================================================")

    p = SingleImageTripoSRPipeline()
    res = p.generate_3d_from_single_image(img_path, num_refine_iterations=100, output_dir=out_dir)

    ply_path = res.get("ply_path")
    if ply_path:
        mesh_paths = export_all_formats(ply_path, out_dir)
        print(f"[OK] Exported 3D Wavefront OBJ model: {mesh_paths['obj']}")
        print(f"[OK] Exported 3D Polygonal PLY mesh:   {mesh_paths['ply_mesh']}")
        print(f"[OK] Exported 3D Standard GLTF asset:  {mesh_paths['gltf']}")

if __name__ == "__main__":
    main()
