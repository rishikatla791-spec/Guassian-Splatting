"""
Point cloud data structure with I/O support for PLY and COLMAP formats.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch


@dataclass
class PointCloud:
    """
    Holds 3D point positions and optional per-point attributes.

    Attributes:
        points: (N, 3) float32 positions in world space
        colors: (N, 3) float32 RGB [0, 1]
        normals: (N, 3) float32 surface normals (optional)
        confidence: (N,) float32 per-point reliability scores (optional)
    """
    points: np.ndarray                            # (N, 3)
    colors: Optional[np.ndarray] = None           # (N, 3)
    normals: Optional[np.ndarray] = None          # (N, 3)
    confidence: Optional[np.ndarray] = None       # (N,)

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=np.float32)
        if self.colors is not None:
            self.colors = np.asarray(self.colors, dtype=np.float32)
            if self.colors.max() > 1.0:
                self.colors = self.colors / 255.0
        if self.normals is not None:
            self.normals = np.asarray(self.normals, dtype=np.float32)
        if self.confidence is not None:
            self.confidence = np.asarray(self.confidence, dtype=np.float32)

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    def filter_by_confidence(self, min_confidence: float) -> "PointCloud":
        """Return new PointCloud with low-confidence points removed."""
        if self.confidence is None:
            return self
        mask = self.confidence >= min_confidence
        return PointCloud(
            points=self.points[mask],
            colors=self.colors[mask] if self.colors is not None else None,
            normals=self.normals[mask] if self.normals is not None else None,
            confidence=self.confidence[mask],
        )

    def remove_statistical_outliers(self, n_neighbors: int = 20, std_ratio: float = 2.0) -> "PointCloud":
        """
        Remove statistical outliers using mean ± std_ratio * std criterion.

        For each point, compute mean distance to n_neighbors nearest neighbors.
        Points with mean_dist > global_mean + std_ratio * global_std are removed.
        """
        pts = torch.tensor(self.points)
        dists = torch.cdist(pts, pts)
        dists.fill_diagonal_(float('inf'))
        knn_dists, _ = dists.topk(n_neighbors, largest=False, dim=-1)  # (N, k)
        mean_dists = knn_dists.mean(dim=-1)  # (N,)

        mu = mean_dists.mean()
        sigma = mean_dists.std()
        keep = (mean_dists <= mu + std_ratio * sigma).numpy()

        return PointCloud(
            points=self.points[keep],
            colors=self.colors[keep] if self.colors is not None else None,
            normals=self.normals[keep] if self.normals is not None else None,
            confidence=self.confidence[keep] if self.confidence is not None else None,
        )

    def to_tensor(self, device: str = "cpu") -> dict:
        """Return dict of torch tensors on given device."""
        result = {"points": torch.tensor(self.points, device=device)}
        if self.colors is not None:
            result["colors"] = torch.tensor(self.colors, device=device)
        if self.normals is not None:
            result["normals"] = torch.tensor(self.normals, device=device)
        return result

    @classmethod
    def from_ply(cls, path: str | Path) -> "PointCloud":
        """Load point cloud from PLY file."""
        from plyfile import PlyData
        plydata = PlyData.read(str(path))
        verts = plydata['vertex']
        points = np.stack([verts['x'], verts['y'], verts['z']], axis=1)

        colors = None
        if all(p in verts.data.dtype.names for p in ['red', 'green', 'blue']):
            colors = np.stack([verts['red'], verts['green'], verts['blue']], axis=1).astype(np.float32)

        normals = None
        if all(p in verts.data.dtype.names for p in ['nx', 'ny', 'nz']):
            normals = np.stack([verts['nx'], verts['ny'], verts['nz']], axis=1)

        return cls(points=points, colors=colors, normals=normals)

    def save_ply(self, path: str | Path) -> None:
        """Save point cloud to PLY file."""
        from plyfile import PlyData, PlyElement

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        has_color = self.colors is not None
        has_normal = self.normals is not None

        attrs = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
        if has_normal:
            attrs += [('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')]
        if has_color:
            attrs += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

        elements = np.empty(self.n_points, dtype=attrs)
        elements['x'] = self.points[:, 0]
        elements['y'] = self.points[:, 1]
        elements['z'] = self.points[:, 2]
        if has_normal:
            elements['nx'] = self.normals[:, 0]
            elements['ny'] = self.normals[:, 1]
            elements['nz'] = self.normals[:, 2]
        if has_color:
            colors_u8 = (np.clip(self.colors, 0, 1) * 255).astype(np.uint8)
            elements['red'] = colors_u8[:, 0]
            elements['green'] = colors_u8[:, 1]
            elements['blue'] = colors_u8[:, 2]

        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(str(path))

    @classmethod
    def from_colmap_points3d(cls, path: str | Path) -> "PointCloud":
        """Load from COLMAP points3D.txt or .bin via binary parsing."""
        path = Path(path)
        if path.suffix == '.bin':
            return cls._from_colmap_bin(path)
        else:
            return cls._from_colmap_txt(path)

    @classmethod
    def _from_colmap_txt(cls, path: Path) -> "PointCloud":
        points, colors = [], []
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                points.append([float(parts[1]), float(parts[2]), float(parts[3])])
                colors.append([int(parts[4]) / 255.0, int(parts[5]) / 255.0, int(parts[6]) / 255.0])
        return cls(points=np.array(points), colors=np.array(colors))

    @classmethod
    def _from_colmap_bin(cls, path: Path) -> "PointCloud":
        import struct
        with open(path, 'rb') as f:
            n_points = struct.unpack('<Q', f.read(8))[0]
            points, colors = [], []
            for _ in range(n_points):
                point3d_id = struct.unpack('<Q', f.read(8))[0]
                xyz = struct.unpack('<3d', f.read(24))
                rgb = struct.unpack('<3B', f.read(3))
                error = struct.unpack('<d', f.read(8))[0]
                track_len = struct.unpack('<Q', f.read(8))[0]
                f.read(8 * track_len)  # skip track (image_id, point2d_idx pairs)
                points.append(list(xyz))
                colors.append([c / 255.0 for c in rgb])
        return cls(points=np.array(points), colors=np.array(colors))

    def __len__(self) -> int:
        return self.n_points

    def __repr__(self) -> str:
        has_c = self.colors is not None
        return f"PointCloud(N={self.n_points}, has_color={has_c})"
