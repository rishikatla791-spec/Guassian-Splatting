"""
Standalone Headless Floater Pruning Module for 3D Gaussian Splatting.

Adapted from gaussian_studio.py::apply_filters:
  - Culls floaters with opacity < min_opacity (default 0.04).
  - Culls oversized Gaussians with max scale > max_scale * scene_radius (default 0.15).
  - Optional ROI bounding sphere crop (crop_radius).
  - Strictly headless: pure NumPy and plyfile, no OpenGL, no GUI, no video rendering.
  - Outputs pruned PLY and optional direct .splat export.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from plyfile import PlyData, PlyElement


class PruneFloatersError(Exception):
    """Exception raised during floater pruning."""
    pass


def compute_scene_radius(xyz: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Computes scene center (median) and robust scene radius (95th percentile distance).
    """
    if len(xyz) == 0:
        return np.zeros(3, dtype=np.float32), 1.0
    center = np.median(xyz, axis=0)
    dists = np.linalg.norm(xyz - center, axis=-1)
    # Use 95th percentile to guard against extreme outlier floaters inflating the radius
    radius = float(np.percentile(dists, 95))
    if radius <= 1e-4:
        radius = float(np.max(dists)) if len(dists) > 0 else 1.0
    return center, max(radius, 1e-3)


def prune_gaussians(
    input_ply: Union[str, Path],
    output_ply: Optional[Union[str, Path]] = None,
    min_opacity: float = 0.04,
    max_scale: Optional[float] = 0.15,
    relative_scale: bool = True,
    crop_radius: Optional[float] = None,
    export_splat: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Filters floaters and oversized Gaussians from a 3DGS PLY file.

    Args:
        input_ply: Path to input point_cloud.ply
        output_ply: Path to save pruned PLY (optional if only exporting splat)
        min_opacity: Opacity threshold in [0, 1]. Gaussians with sigmoid(opacity) < min_opacity are culled.
        max_scale: Maximum allowed scale. If relative_scale is True, threshold is max_scale * scene_radius.
        relative_scale: Whether max_scale is relative to scene radius.
        crop_radius: Optional ROI crop radius around scene center.
        export_splat: Path to export 32-byte binary .splat file (optional).

    Returns:
        Summary dict of pruning statistics.
    """
    input_ply = Path(input_ply).resolve()
    if not input_ply.exists():
        raise PruneFloatersError(f"Input PLY file not found: {input_ply}")

    try:
        plydata = PlyData.read(str(input_ply))
    except Exception as e:
        raise PruneFloatersError(f"Failed to parse PLY file {input_ply}: {e}")

    if 'vertex' not in plydata:
        raise PruneFloatersError(f"PLY file {input_ply} does not contain 'vertex' element")

    vertex = plydata['vertex'].data
    total_count = len(vertex)
    if total_count == 0:
        raise PruneFloatersError(f"PLY file {input_ply} contains 0 vertices")

    # 1. Coordinates and Scene Extent
    x = np.asarray(vertex['x'], dtype=np.float32)
    y = np.asarray(vertex['y'], dtype=np.float32)
    z = np.asarray(vertex['z'], dtype=np.float32)
    xyz = np.column_stack([x, y, z])

    center, scene_radius = compute_scene_radius(xyz)

    # Initialize keep mask
    mask = np.ones(total_count, dtype=bool)

    # 2. Opacity Filtering: sigmoid(opacity) >= min_opacity
    culled_opacity = 0
    if 'opacity' in vertex.dtype.names and min_opacity > 0.0:
        raw_opacity = np.asarray(vertex['opacity'], dtype=np.float32)
        # Numerical stability for sigmoid
        clipped_op = np.clip(raw_opacity, -20.0, 20.0)
        opacity = 1.0 / (1.0 + np.exp(-clipped_op))
        opacity_mask = (opacity >= min_opacity)
        culled_opacity = int(np.sum(~opacity_mask))
        mask = mask & opacity_mask

    # 3. Scale Filtering: max(exp(scale_i)) <= threshold
    culled_scale = 0
    scale_threshold = None
    if max_scale is not None and max_scale > 0.0:
        if 'scale_0' in vertex.dtype.names and 'scale_1' in vertex.dtype.names and 'scale_2' in vertex.dtype.names:
            s0 = np.exp(np.clip(np.asarray(vertex['scale_0'], dtype=np.float32), -20.0, 20.0))
            s1 = np.exp(np.clip(np.asarray(vertex['scale_1'], dtype=np.float32), -20.0, 20.0))
            s2 = np.exp(np.clip(np.asarray(vertex['scale_2'], dtype=np.float32), -20.0, 20.0))
            max_s = np.maximum(np.maximum(s0, s1), s2)

            if relative_scale:
                scale_threshold = float(max_scale * scene_radius)
            else:
                scale_threshold = float(max_scale)

            scale_mask = (max_s <= scale_threshold)
            culled_scale = int(np.sum(~scale_mask))
            mask = mask & scale_mask

    # 4. ROI Cropping
    if crop_radius is not None and crop_radius > 0.0:
        dists = np.linalg.norm(xyz - center, axis=-1)
        crop_mask = (dists <= crop_radius)
        mask = mask & crop_mask

    kept_count = int(np.sum(mask))
    culled_count = total_count - kept_count
    kept_pct = (kept_count / total_count * 100.0) if total_count > 0 else 0.0

    if kept_count == 0:
        # Fallback safeguard: if all Gaussians were filtered out, keep top 10% highest opacity
        if 'opacity' in vertex.dtype.names:
            raw_op = np.asarray(vertex['opacity'], dtype=np.float32)
            k = max(1, int(total_count * 0.1))
            top_idx = np.argpartition(raw_op, -k)[-k:]
            mask[top_idx] = True
            kept_count = int(np.sum(mask))
            culled_count = total_count - kept_count
            kept_pct = (kept_count / total_count * 100.0)

    # 5. Output Pruned PLY
    filtered_vertex = vertex[mask]
    if output_ply is not None:
        output_ply = Path(output_ply).resolve()
        output_ply.parent.mkdir(parents=True, exist_ok=True)
        el = PlyElement.describe(filtered_vertex, 'vertex')
        PlyData([el], text=False).write(str(output_ply))

    # 6. Optional Direct SPLAT Export
    if export_splat is not None:
        export_splat = Path(export_splat).resolve()
        export_splat.parent.mkdir(parents=True, exist_ok=True)
        _export_to_splat_binary(filtered_vertex, export_splat)

    stats = {
        "status": "success",
        "input_path": str(input_ply),
        "output_path": str(output_ply) if output_ply else None,
        "export_splat_path": str(export_splat) if export_splat else None,
        "initial_gaussians": total_count,
        "final_gaussians": kept_count,
        "kept_gaussians": kept_count,
        "culled_gaussians": culled_count,
        "culled_total": culled_count,
        "culled_opacity": culled_opacity,
        "culled_scale": culled_scale,
        "kept_percentage": round(kept_pct, 2),
        "scene_radius": round(scene_radius, 4),
        "min_opacity_threshold": min_opacity,
        "max_scale_threshold": round(scale_threshold, 6) if scale_threshold else None,
    }

    return stats


def _export_to_splat_binary(vertex: np.ndarray, splat_path: Path) -> None:
    """
    Direct high-throughput vector conversion of PLY vertex elements to 32-byte .splat.
    """
    n = len(vertex)
    x = np.asarray(vertex['x'], dtype=np.float32)
    y = np.asarray(vertex['y'], dtype=np.float32)
    z = np.asarray(vertex['z'], dtype=np.float32)

    s0 = np.exp(np.clip(np.asarray(vertex['scale_0'], dtype=np.float32), -20.0, 20.0))
    s1 = np.exp(np.clip(np.asarray(vertex['scale_1'], dtype=np.float32), -20.0, 20.0))
    s2 = np.exp(np.clip(np.asarray(vertex['scale_2'], dtype=np.float32), -20.0, 20.0))

    C0 = 0.28209479177387814
    f_dc_0 = np.asarray(vertex['f_dc_0'], dtype=np.float32)
    f_dc_1 = np.asarray(vertex['f_dc_1'], dtype=np.float32)
    f_dc_2 = np.asarray(vertex['f_dc_2'], dtype=np.float32)

    r = np.clip((0.5 + C0 * f_dc_0) * 255.0, 0, 255).astype(np.uint8)
    g = np.clip((0.5 + C0 * f_dc_1) * 255.0, 0, 255).astype(np.uint8)
    b = np.clip((0.5 + C0 * f_dc_2) * 255.0, 0, 255).astype(np.uint8)

    raw_op = np.asarray(vertex['opacity'], dtype=np.float32)
    op = (1.0 / (1.0 + np.exp(-np.clip(raw_op, -20.0, 20.0)))) * 255.0
    a = np.clip(op, 0, 255).astype(np.uint8)

    rot_0 = np.asarray(vertex['rot_0'], dtype=np.float32)
    rot_1 = np.asarray(vertex['rot_1'], dtype=np.float32)
    rot_2 = np.asarray(vertex['rot_2'], dtype=np.float32)
    rot_3 = np.asarray(vertex['rot_3'], dtype=np.float32)

    norm = np.sqrt(rot_0**2 + rot_1**2 + rot_2**2 + rot_3**2) + 1e-8
    rot_0 = rot_0 / norm
    rot_1 = rot_1 / norm
    rot_2 = rot_2 / norm
    rot_3 = rot_3 / norm

    q0 = np.clip(rot_0 * 128.0 + 128.0, 0, 255).astype(np.uint8)
    q1 = np.clip(rot_1 * 128.0 + 128.0, 0, 255).astype(np.uint8)
    q2 = np.clip(rot_2 * 128.0 + 128.0, 0, 255).astype(np.uint8)
    q3 = np.clip(rot_3 * 128.0 + 128.0, 0, 255).astype(np.uint8)

    splat_dtype = np.dtype([
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('s0', '<f4'), ('s1', '<f4'), ('s2', '<f4'),
        ('r', 'u1'), ('g', 'u1'), ('b', 'u1'), ('a', 'u1'),
        ('q0', 'u1'), ('q1', 'u1'), ('q2', 'u1'), ('q3', 'u1')
    ])
    arr = np.empty(n, dtype=splat_dtype)
    arr['x'] = x
    arr['y'] = y
    arr['z'] = z
    arr['s0'] = s0
    arr['s1'] = s1
    arr['s2'] = s2
    arr['r'] = r
    arr['g'] = g
    arr['b'] = b
    arr['a'] = a
    arr['q0'] = q0
    arr['q1'] = q1
    arr['q2'] = q2
    arr['q3'] = q3

    arr.tofile(str(splat_path))


def main():
    parser = argparse.ArgumentParser(description="Standalone Headless 3D Gaussian Floater Pruner")
    parser.add_argument("--input", "-i", required=True, type=str, help="Input PLY file path")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output pruned PLY file path")
    parser.add_argument("--min_opacity", type=float, default=0.04, help="Min sigmoid opacity threshold (default: 0.04)")
    parser.add_argument("--max_scale", type=float, default=0.15, help="Max scale threshold (default: 0.15 of scene radius)")
    parser.add_argument("--absolute_scale", action="store_true", help="Treat max_scale as absolute instead of relative to scene radius")
    parser.add_argument("--crop_radius", type=float, default=None, help="Optional ROI crop radius")
    parser.add_argument("--export_splat", type=str, default=None, help="Optional direct .splat export path")

    args = parser.parse_args()

    if not args.output and not args.export_splat:
        parser.error("At least one of --output or --export_splat must be specified")

    try:
        stats = prune_gaussians(
            input_ply=args.input,
            output_ply=args.output,
            min_opacity=args.min_opacity,
            max_scale=args.max_scale,
            relative_scale=not args.absolute_scale,
            crop_radius=args.crop_radius,
            export_splat=args.export_splat
        )
        print(json.dumps(stats, indent=2))
        sys.exit(0)
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
