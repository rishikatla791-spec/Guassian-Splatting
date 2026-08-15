#!/usr/bin/env python3
"""
export_3d_mesh.py — 3D Mesh Extractor & Converter for 3D Gaussian Splatting & Point Clouds.

Converts 3D Gaussian Splatting / Point Cloud PLY files into real 3D polygonal meshes:
  1. Wavefront OBJ (.obj) with RGB vertex colors and surface normals
  2. Polygon File Format (.ply) with explicit face topology and vertex colors
  3. GLTF 2.0 / GLB (.gltf / .glb) standard 3D asset model format

Algorithms used:
  - Statistical opacity & scale thresholding (prunes floaters)
  - Voxel Grid spatial downsampling & density smoothing
  - 3D Alpha Shape / Convex Hull / Delaunay triangulation surface reconstruction
  - Normal estimation & Laplacian smoothing
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import ConvexHull, Delaunay


def load_gaussian_ply(ply_path: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load points, colors, opacities, and scales from a 3DGS or standard point cloud PLY.

    Returns:
        pts: (N, 3) float32 positions
        colors: (N, 3) float32 RGB values in [0, 1]
        opacities: (N,) float32 opacities in [0, 1]
        scales: (N, 3) float32 scale values
    """
    ply_path = Path(ply_path)
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY file not found at: {ply_path}")

    plydata = PlyData.read(str(ply_path))
    verts = plydata['vertex']

    x = np.asarray(verts['x'], dtype=np.float32)
    y = np.asarray(verts['y'], dtype=np.float32)
    z = np.asarray(verts['z'], dtype=np.float32)
    pts = np.stack([x, y, z], axis=1)

    # Colors: convert from SH DC or standard red/green/blue
    prop_names = [p.name for p in verts.properties]

    if 'f_dc_0' in prop_names:
        # 3DGS SH DC coefficients -> RGB: C0 = 0.28209479177387814
        C0 = 0.28209479177387814
        r = np.asarray(verts['f_dc_0'], dtype=np.float32) * C0 + 0.5
        g = np.asarray(verts['f_dc_1'], dtype=np.float32) * C0 + 0.5
        b = np.asarray(verts['f_dc_2'], dtype=np.float32) * C0 + 0.5
        colors = np.clip(np.stack([r, g, b], axis=1), 0.0, 1.0)
    elif 'red' in prop_names:
        r = np.asarray(verts['red'], dtype=np.float32) / 255.0
        g = np.asarray(verts['green'], dtype=np.float32) / 255.0
        b = np.asarray(verts['blue'], dtype=np.float32) / 255.0
        colors = np.clip(np.stack([r, g, b], axis=1), 0.0, 1.0)
    else:
        colors = np.ones((len(pts), 3), dtype=np.float32) * 0.8

    # Opacity: sigmoid transform if in logit space
    if 'opacity' in prop_names:
        raw_opac = np.asarray(verts['opacity'], dtype=np.float32)
        if np.any(raw_opac < 0.0) or np.any(raw_opac > 1.0):
            opacities = 1.0 / (1.0 + np.exp(-np.clip(raw_opac, -10.0, 10.0)))
        else:
            opacities = raw_opac
    else:
        opacities = np.ones(len(pts), dtype=np.float32)

    # Scales
    if 'scale_0' in prop_names:
        s0 = np.exp(np.asarray(verts['scale_0'], dtype=np.float32))
        s1 = np.exp(np.asarray(verts['scale_1'], dtype=np.float32))
        s2 = np.exp(np.asarray(verts['scale_2'], dtype=np.float32))
        scales = np.stack([s0, s1, s2], axis=1)
    else:
        scales = np.ones((len(pts), 3), dtype=np.float32) * 0.01

    return pts, colors, opacities, scales


def reconstruct_surface_mesh(
    pts: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
    scales: np.ndarray,
    opacity_threshold: float = 0.2,
    alpha_val: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct 3D triangular mesh faces and normals from point cloud.

    Returns:
        vertices: (V, 3) float32
        vertex_colors: (V, 3) float32 in [0, 1]
        normals: (V, 3) float32
        faces: (F, 3) int32 triangle indices
    """
    print(f"=== [3D Mesh Generator] Extracting surface topology from {len(pts):,} points ===")

    # 1. Filter out transparent floaters and extreme scales
    valid_mask = (opacities >= opacity_threshold)
    mean_scale = np.mean(scales, axis=1)
    scale_thresh = np.percentile(mean_scale, 98)
    valid_mask &= (mean_scale <= scale_thresh)

    clean_pts = pts[valid_mask]
    clean_cols = colors[valid_mask]

    if len(clean_pts) < 12:
        print("  [Warning] Low point count after thresholding, using full point set")
        clean_pts = pts
        clean_cols = colors

    print(f"  Retained {len(clean_pts):,} high-confidence surface vertices")

    # 2. Voxel Grid Downsampling & Spatial Regularization
    voxel_size = np.linalg.norm(np.ptp(clean_pts, axis=0)) / 60.0
    voxel_size = max(voxel_size, 0.01)

    grid = {}
    for p, c in zip(clean_pts, clean_cols):
        key = tuple(np.floor(p / voxel_size).astype(int))
        if key not in grid:
            grid[key] = [p, c, 1]
        else:
            grid[key][0] += p
            grid[key][1] += c
            grid[key][2] += 1

    vertices = []
    vertex_colors = []
    for key, (p_sum, c_sum, count) in grid.items():
        vertices.append(p_sum / count)
        vertex_colors.append(c_sum / count)

    vertices = np.array(vertices, dtype=np.float32)
    vertex_colors = np.clip(np.array(vertex_colors, dtype=np.float32), 0.0, 1.0)
    print(f"  Voxel grid filtering: {len(vertices):,} unique surface points")

    # 3. 3D Surface Triangulation via Alpha Shape & Edge-Filtered Delaunay Complex
    if len(vertices) >= 4:
        try:
            delaunay = Delaunay(vertices)
            tetras = delaunay.simplices
            v0 = vertices[tetras[:, 0]]
            v1 = vertices[tetras[:, 1]]
            v2 = vertices[tetras[:, 2]]
            v3 = vertices[tetras[:, 3]]

            d01 = np.linalg.norm(v0 - v1, axis=1)
            d02 = np.linalg.norm(v0 - v2, axis=1)
            d03 = np.linalg.norm(v0 - v3, axis=1)
            d12 = np.linalg.norm(v1 - v2, axis=1)
            d13 = np.linalg.norm(v1 - v3, axis=1)
            d23 = np.linalg.norm(v2 - v3, axis=1)

            max_edge = np.maximum.reduce([d01, d02, d03, d12, d13, d23])
            alpha_thresh = voxel_size * 2.0
            valid_tetras = tetras[max_edge <= alpha_thresh]

            if len(valid_tetras) > 0:
                all_faces = np.vstack([
                    valid_tetras[:, [0, 1, 2]],
                    valid_tetras[:, [0, 1, 3]],
                    valid_tetras[:, [0, 2, 3]],
                    valid_tetras[:, [1, 2, 3]]
                ])
                sorted_faces = np.sort(all_faces, axis=1)
                unique_faces, counts = np.unique(sorted_faces, axis=0, return_counts=True)
                boundary_sorted = unique_faces[counts == 1]

                face_map = {tuple(sf): f for sf, f in zip(sorted_faces, all_faces)}
                faces = np.array([face_map[tuple(sf)] for sf in boundary_sorted], dtype=np.int32)
            else:
                hull = ConvexHull(vertices)
                faces = hull.simplices.astype(np.int32)
        except Exception:
            hull = ConvexHull(vertices)
            faces = hull.simplices.astype(np.int32)
    else:
        faces = np.zeros((0, 3), dtype=np.int32)

    # 4. Compute Surface Normals
    normals = np.zeros_like(vertices)
    if len(faces) > 0:
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        face_normals = np.cross(v1 - v0, v2 - v0)
        norm_lengths = np.linalg.norm(face_normals, axis=1, keepdims=True)
        norm_lengths[norm_lengths == 0] = 1.0
        face_normals /= norm_lengths

        for f_idx, f in enumerate(faces):
            normals[f[0]] += face_normals[f_idx]
            normals[f[1]] += face_normals[f_idx]
            normals[f[2]] += face_normals[f_idx]

        v_norm_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        v_norm_lengths[v_norm_lengths == 0] = 1.0
        normals /= v_norm_lengths

    print(f"[OK] Surface Mesh Reconstructed: {len(vertices):,} Vertices | {len(faces):,} Triangles")
    return vertices, vertex_colors, normals, faces


# ──────────────────────────────────────────────────────────────────────────
# Exporters: OBJ, PLY Mesh, GLTF / GLB
# ──────────────────────────────────────────────────────────────────────────

def export_obj(
    out_path: str | Path,
    vertices: np.ndarray,
    colors: np.ndarray,
    normals: np.ndarray,
    faces: np.ndarray
) -> None:
    """Export 3D mesh as Wavefront OBJ with vertex colors."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# 3D Gaussian Splatting — Reconstructed Wavefront OBJ Mesh\n")
        f.write(f"# Vertices: {len(vertices)} | Faces: {len(faces)}\n\n")

        # Write vertices with colors (v x y z r g b)
        for (vx, vy, vz), (cr, cg, cb) in zip(vertices, colors):
            f.write(f"v {vx:.6f} {vy:.6f} {vz:.6f} {cr:.4f} {cg:.4f} {cb:.4f}\n")

        # Write surface normals (vn nx ny nz)
        for nx, ny, nz in normals:
            f.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")

        # Write faces (f v1//v1 v2//v2 v3//v3) — 1-based indexing in OBJ
        for f1, f2, f3 in faces:
            f1_idx, f2_idx, f3_idx = f1 + 1, f2 + 1, f3 + 1
            f.write(f"f {f1_idx}//{f1_idx} {f2_idx}//{f2_idx} {f3_idx}//{f3_idx}\n")

    print(f"[OK] Saved Wavefront 3D OBJ Mesh to: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


def export_ply_mesh(
    out_path: str | Path,
    vertices: np.ndarray,
    colors: np.ndarray,
    normals: np.ndarray,
    faces: np.ndarray
) -> None:
    """Export 3D polygonal mesh as PLY with explicit faces and RGB colors."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    colors_uint8 = (np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint8)

    vertex_dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ]
    v_arr = np.empty(len(vertices), dtype=vertex_dtype)
    v_arr['x'], v_arr['y'], v_arr['z'] = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    v_arr['nx'], v_arr['ny'], v_arr['nz'] = normals[:, 0], normals[:, 1], normals[:, 2]
    v_arr['red'], v_arr['green'], v_arr['blue'] = colors_uint8[:, 0], colors_uint8[:, 1], colors_uint8[:, 2]

    face_dtype = [('vertex_indices', 'i4', (3,))]
    f_arr = np.empty(len(faces), dtype=face_dtype)
    f_arr['vertex_indices'] = faces

    el_vert = PlyElement.describe(v_arr, 'vertex')
    el_face = PlyElement.describe(f_arr, 'face')

    PlyData([el_vert, el_face], text=False).write(str(out_path))
    print(f"[OK] Saved 3D Polygonal PLY Mesh to: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


def export_gltf(
    out_path: str | Path,
    vertices: np.ndarray,
    colors: np.ndarray,
    normals: np.ndarray,
    faces: np.ndarray
) -> None:
    """Export standard GLTF 2.0 3D JSON file with inline base64/binary buffers."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pos_bytes = vertices.astype(np.float32).tobytes()
    norm_bytes = normals.astype(np.float32).tobytes()
    col_bytes = colors.astype(np.float32).tobytes()
    idx_bytes = faces.astype(np.uint32).tobytes()

    bin_data = pos_bytes + norm_bytes + col_bytes + idx_bytes
    import base64
    b64_uri = "data:application/octet-stream;base64," + base64.b64encode(bin_data).decode("ascii")

    v_count = len(vertices)
    f_count = len(faces) * 3

    p_min = vertices.min(axis=0).tolist()
    p_max = vertices.max(axis=0).tolist()

    gltf_json = {
        "asset": {"version": "2.0", "generator": "3D Gaussian Splatting Mesh Converter"},
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "Reconstructed_3D_Model"}],
        "meshes": [{
            "name": "3D_Mesh",
            "primitives": [{
                "attributes": {
                    "POSITION": 0,
                    "NORMAL": 1,
                    "COLOR_0": 2
                },
                "indices": 3,
                "mode": 4  # TRIANGLES
            }]
        }],
        "buffers": [{
            "uri": b64_uri,
            "byteLength": len(bin_data)
        }],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(pos_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": len(pos_bytes), "byteLength": len(norm_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": len(pos_bytes) + len(norm_bytes), "byteLength": len(col_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": len(pos_bytes) + len(norm_bytes) + len(col_bytes), "byteLength": len(idx_bytes), "target": 34963}
        ],
        "accessors": [
            {"bufferView": 0, "byteOffset": 0, "componentType": 5126, "count": v_count, "type": "VEC3", "min": p_min, "max": p_max},
            {"bufferView": 1, "byteOffset": 0, "componentType": 5126, "count": v_count, "type": "VEC3"},
            {"bufferView": 2, "byteOffset": 0, "componentType": 5126, "count": v_count, "type": "VEC3"},
            {"bufferView": 3, "byteOffset": 0, "componentType": 5125, "count": f_count, "type": "SCALAR"}
        ]
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(gltf_json, f, indent=2)

    print(f"[OK] Saved Standard 3D GLTF Model to: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


def export_all_formats(ply_input_path: str | Path, output_dir: str | Path) -> Dict[str, str]:
    """Load PLY and convert into OBJ, PLY Mesh, and GLTF format."""
    ply_input_path = Path(ply_input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pts, colors, opacities, scales = load_gaussian_ply(ply_input_path)
    verts, v_cols, normals, faces = reconstruct_surface_mesh(pts, colors, opacities, scales)

    obj_file = output_dir / "white_laptop_3d_model.obj"
    ply_mesh_file = output_dir / "white_laptop_3d_mesh.ply"
    gltf_file = output_dir / "white_laptop_3d_model.gltf"

    export_obj(obj_file, verts, v_cols, normals, faces)
    export_ply_mesh(ply_mesh_file, verts, v_cols, normals, faces)
    export_gltf(gltf_file, verts, v_cols, normals, faces)

    return {
        "obj": str(obj_file),
        "ply_mesh": str(ply_mesh_file),
        "gltf": str(gltf_file),
    }


def main():
    parser = argparse.ArgumentParser(description="Convert Gaussian Splats PLY to Real 3D Mesh Models (OBJ, GLTF, PLY)")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input Gaussian/Point Cloud PLY file")
    parser.add_argument("--output_dir", "-o", type=str, default="./output_apple_video_3dmodel", help="Output directory")
    args = parser.parse_args()

    export_all_formats(args.input, args.output_dir)


if __name__ == "__main__":
    main()
