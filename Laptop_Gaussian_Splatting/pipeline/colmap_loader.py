"""
ColmapSceneLoader: Full binary and text COLMAP sparse reconstruction reader.

Binary format references:
  https://colmap.github.io/format.html

Supported camera models (COLMAP model IDs):
  0  SIMPLE_PINHOLE  — params: f, cx, cy
  1  PINHOLE         — params: fx, fy, cx, cy
  2  SIMPLE_RADIAL   — params: f, cx, cy, k1
  3  RADIAL          — params: f, cx, cy, k1, k2
  4  OPENCV          — params: fx, fy, cx, cy, k1, k2, p1, p2
  5  FULL_OPENCV     — params: fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6

Coordinate conventions (COLMAP):
  World space: right-handed (x right, y down, z into scene)
  q = [w, x, y, z] (Hamilton convention)
  R  = quat_to_rotation(q)           # world → camera rotation
  t  = translation vector (world → camera)
  Camera center in world: C = −Rᵀ t
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from gaussian.core.camera import Camera, CameraIntrinsics, CameraExtrinsics
except ImportError:
    from core.camera import Camera, CameraIntrinsics, CameraExtrinsics


# ---------------------------------------------------------------------------
# COLMAP camera model definitions
# ---------------------------------------------------------------------------

#: Map model_id → (model_name, num_params)
CAMERA_MODELS: Dict[int, Tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE",        4),
    2: ("SIMPLE_RADIAL",  4),
    3: ("RADIAL",         5),
    4: ("OPENCV",         8),
    5: ("FULL_OPENCV",    12),
    6: ("SIMPLE_RADIAL_FISHEYE", 4),
    7: ("RADIAL_FISHEYE", 5),
    8: ("OPENCV_FISHEYE", 8),
    9: ("FOV",            5),
    10: ("THIN_PRISM_FISHEYE", 12),
}

# Model ID → name lookup (also by name → id)
CAMERA_MODEL_IDS: Dict[str, int] = {v[0]: k for k, v in CAMERA_MODELS.items()}


# ---------------------------------------------------------------------------
# Raw COLMAP data containers (not exposed publicly)
# ---------------------------------------------------------------------------

class _ColmapCamera:
    """Raw COLMAP camera record (before conversion to CameraIntrinsics)."""
    __slots__ = ("camera_id", "model_id", "model_name", "width", "height", "params")

    def __init__(
        self,
        camera_id: int,
        model_id: int,
        width: int,
        height: int,
        params: np.ndarray,
    ) -> None:
        self.camera_id = camera_id
        self.model_id  = model_id
        self.model_name, _ = CAMERA_MODELS.get(model_id, ("UNKNOWN", 0))
        self.width  = width
        self.height = height
        self.params = params  # float64 array of intrinsic parameters


class _ColmapImage:
    """Raw COLMAP image record."""
    __slots__ = (
        "image_id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"
    )

    def __init__(
        self,
        image_id: int,
        qvec: np.ndarray,   # [w, x, y, z]
        tvec: np.ndarray,   # [tx, ty, tz]
        camera_id: int,
        name: str,
        xys: np.ndarray,        # (M, 2) 2D keypoints
        point3D_ids: np.ndarray, # (M,) int64 matching point3D ids (−1 = unmatched)
    ) -> None:
        self.image_id    = image_id
        self.qvec        = qvec
        self.tvec        = tvec
        self.camera_id   = camera_id
        self.name        = name
        self.xys         = xys
        self.point3D_ids = point3D_ids


class _ColmapPoint3D:
    """Raw COLMAP Point3D record."""
    __slots__ = ("point3d_id", "xyz", "rgb", "error", "track")

    def __init__(
        self,
        point3d_id: int,
        xyz: np.ndarray,   # (3,) float64
        rgb: np.ndarray,   # (3,) uint8
        error: float,
        track: np.ndarray, # (T, 2) int32 [(image_id, point2D_idx), ...]
    ) -> None:
        self.point3d_id = point3d_id
        self.xyz        = xyz
        self.rgb        = rgb
        self.error      = error
        self.track      = track


# ---------------------------------------------------------------------------
# Binary readers
# ---------------------------------------------------------------------------

def _read_cameras_binary(path: str) -> Dict[int, _ColmapCamera]:
    """Parse cameras.bin → {camera_id: _ColmapCamera}."""
    cameras: Dict[int, _ColmapCamera] = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]  # uint64
        for _ in range(num_cameras):
            camera_id = struct.unpack("<i", f.read(4))[0]   # int32
            model_id  = struct.unpack("<i", f.read(4))[0]   # int32
            width     = struct.unpack("<Q", f.read(8))[0]   # uint64
            height    = struct.unpack("<Q", f.read(8))[0]   # uint64

            _, num_params = CAMERA_MODELS.get(model_id, ("UNKNOWN", 0))
            params = np.array(
                struct.unpack(f"<{num_params}d", f.read(8 * num_params)),
                dtype=np.float64,
            )
            cameras[camera_id] = _ColmapCamera(camera_id, model_id, width, height, params)
    return cameras


def _read_images_binary(path: str) -> Dict[int, _ColmapImage]:
    """Parse images.bin → {image_id: _ColmapImage}."""
    images: Dict[int, _ColmapImage] = {}
    with open(path, "rb") as f:
        num_reg_images = struct.unpack("<Q", f.read(8))[0]  # uint64
        for _ in range(num_reg_images):
            image_id  = struct.unpack("<i", f.read(4))[0]   # int32
            qvec      = np.array(struct.unpack("<4d", f.read(32)), dtype=np.float64)  # w,x,y,z
            tvec      = np.array(struct.unpack("<3d", f.read(24)), dtype=np.float64)  # tx,ty,tz
            camera_id = struct.unpack("<i", f.read(4))[0]   # int32

            # Null-terminated name string
            name_chars: List[bytes] = []
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_chars.append(c)
            name = b"".join(name_chars).decode("utf-8")

            num_points2D = struct.unpack("<Q", f.read(8))[0]  # uint64
            # Each point2D: x (double), y (double), point3D_id (int64)
            xys_list: List[Tuple[float, float]] = []
            p3d_ids: List[int] = []
            for _p in range(num_points2D):
                x, y   = struct.unpack("<2d", f.read(16))
                p3d_id = struct.unpack("<q", f.read(8))[0]  # int64
                xys_list.append((x, y))
                p3d_ids.append(p3d_id)

            xys         = np.array(xys_list, dtype=np.float64).reshape(-1, 2)
            point3D_ids = np.array(p3d_ids, dtype=np.int64)

            images[image_id] = _ColmapImage(
                image_id, qvec, tvec, camera_id, name, xys, point3D_ids
            )
    return images


def _read_points3d_binary(path: str) -> Dict[int, _ColmapPoint3D]:
    """Parse points3D.bin → {point3d_id: _ColmapPoint3D}."""
    points: Dict[int, _ColmapPoint3D] = {}
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]  # uint64
        for _ in range(num_points):
            point3d_id = struct.unpack("<Q", f.read(8))[0]  # uint64
            xyz   = np.array(struct.unpack("<3d", f.read(24)), dtype=np.float64)
            rgb   = np.array(struct.unpack("<3B", f.read(3)),  dtype=np.uint8)
            error = struct.unpack("<d", f.read(8))[0]  # float64
            track_length = struct.unpack("<Q", f.read(8))[0]  # uint64
            track_raw = struct.unpack(f"<{2 * track_length}i", f.read(8 * track_length))
            track = np.array(track_raw, dtype=np.int32).reshape(-1, 2)  # (T, 2)
            points[point3d_id] = _ColmapPoint3D(point3d_id, xyz, rgb, error, track)
    return points


# ---------------------------------------------------------------------------
# Text readers (fallback)
# ---------------------------------------------------------------------------

def _read_cameras_text(path: str) -> Dict[int, _ColmapCamera]:
    """Parse cameras.txt → {camera_id: _ColmapCamera}."""
    cameras: Dict[int, _ColmapCamera] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            camera_id  = int(parts[0])
            model_name = parts[1]
            width      = int(parts[2])
            height     = int(parts[3])
            params     = np.array([float(p) for p in parts[4:]], dtype=np.float64)
            model_id   = CAMERA_MODEL_IDS.get(model_name, -1)
            cameras[camera_id] = _ColmapCamera(camera_id, model_id, width, height, params)
    return cameras


def _read_images_text(path: str) -> Dict[int, _ColmapImage]:
    """Parse images.txt → {image_id: _ColmapImage}."""
    images: Dict[int, _ColmapImage] = {}
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    # Lines come in pairs: metadata line, then point2D line
    i = 0
    while i < len(lines):
        meta_parts = lines[i].split()
        image_id  = int(meta_parts[0])
        qvec      = np.array([float(meta_parts[j]) for j in range(1, 5)], dtype=np.float64)
        tvec      = np.array([float(meta_parts[j]) for j in range(5, 8)], dtype=np.float64)
        camera_id = int(meta_parts[8])
        name      = meta_parts[9]

        i += 1
        if i < len(lines):
            pts_parts = lines[i].split()
        else:
            pts_parts = []

        xys_list: List[Tuple[float, float]] = []
        p3d_ids:  List[int] = []
        j = 0
        while j + 2 < len(pts_parts):
            xys_list.append((float(pts_parts[j]), float(pts_parts[j + 1])))
            p3d_ids.append(int(pts_parts[j + 2]))
            j += 3

        xys         = np.array(xys_list, dtype=np.float64).reshape(-1, 2)
        point3D_ids = np.array(p3d_ids, dtype=np.int64)

        images[image_id] = _ColmapImage(
            image_id, qvec, tvec, camera_id, name, xys, point3D_ids
        )
        i += 1
    return images


def _read_points3d_text(path: str) -> Dict[int, _ColmapPoint3D]:
    """Parse points3D.txt → {point3d_id: _ColmapPoint3D}."""
    points: Dict[int, _ColmapPoint3D] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            point3d_id = int(parts[0])
            xyz   = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
            rgb   = np.array([int(parts[4]),   int(parts[5]),   int(parts[6])],   dtype=np.uint8)
            error = float(parts[7])
            # Remaining: alternating image_id, point2D_idx pairs
            track_vals = [int(p) for p in parts[8:]]
            if track_vals:
                track = np.array(track_vals, dtype=np.int32).reshape(-1, 2)
            else:
                track = np.zeros((0, 2), dtype=np.int32)
            points[point3d_id] = _ColmapPoint3D(point3d_id, xyz, rgb, error, track)
    return points


# ---------------------------------------------------------------------------
# Quaternion → rotation matrix (NumPy)
# ---------------------------------------------------------------------------

def _qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """
    Convert COLMAP quaternion [w, x, y, z] to 3×3 rotation matrix.

    R maps world → camera.
    """
    w, x, y, z = qvec / np.linalg.norm(qvec)
    R = np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x*x + z*z),  2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),      1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    return R


# ---------------------------------------------------------------------------
# Intrinsics factory
# ---------------------------------------------------------------------------

def _make_intrinsics(
    cam: _ColmapCamera,
    target_resolution: int,
) -> Tuple[CameraIntrinsics, int, int]:
    """
    Build CameraIntrinsics from a raw _ColmapCamera.

    Handles portrait/landscape orientation detection and optional resolution
    downscaling while preserving aspect ratio and adjusting focal lengths.

    Args:
        cam:               raw COLMAP camera
        target_resolution: if > 0 the longer edge is scaled to this value;
                           if −1 the resolution is kept as-is.

    Returns:
        (intrinsics, final_width, final_height)
    """
    p = cam.params
    model = cam.model_name

    # --- Extract focal lengths and principal point -------------------------
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"):
        fx = fy = float(p[0])
        cx, cy  = float(p[1]), float(p[2])
    elif model in ("PINHOLE", "RADIAL", "OPENCV", "FULL_OPENCV",
                   "RADIAL_FISHEYE", "OPENCV_FISHEYE", "THIN_PRISM_FISHEYE"):
        fx, fy  = float(p[0]), float(p[1])
        cx, cy  = float(p[2]), float(p[3])
    elif model == "FOV":
        fx, fy  = float(p[0]), float(p[1])
        cx, cy  = float(p[2]), float(p[3])
    else:
        # Fallback: assume PINHOLE layout
        fx = fy = float(p[0])
        cx = cam.width  / 2.0
        cy = cam.height / 2.0

    width  = int(cam.width)
    height = int(cam.height)

    # --- Portrait / landscape orientation detection -----------------------
    # COLMAP always stores images in the orientation they were registered.
    # We keep width/height as-is; the Camera object stores these verbatim.

    # --- Resolution scaling -----------------------------------------------
    if target_resolution > 0:
        # Scale so that the longer edge == target_resolution
        longer_edge = max(width, height)
        if longer_edge != target_resolution:
            scale = target_resolution / longer_edge
            new_width  = int(round(width  * scale))
            new_height = int(round(height * scale))
            # Adjust focal lengths and principal point proportionally
            fx *= (new_width  / width)
            fy *= (new_height / height)
            cx *= (new_width  / width)
            cy *= (new_height / height)
            width, height = new_width, new_height

    return (
        CameraIntrinsics(
            fx=fx, fy=fy, cx=cx, cy=cy,
            width=width, height=height,
        ),
        width,
        height,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ColmapSceneLoader:
    """
    Loads a COLMAP sparse reconstruction and converts it to Camera objects
    and a colored point cloud for use in 3D Gaussian Splatting.

    Usage::

        loader = ColmapSceneLoader()
        cameras, points3d, colors = loader.load(
            colmap_path="path/to/sparse/0",
            images_path="path/to/images",
            resolution=1920,
        )
    """

    # ------------------------------------------------------------------
    # Construction helpers (static)
    # ------------------------------------------------------------------

    @staticmethod
    def _locate_sparse(colmap_path: str) -> Tuple[bool, str, str, str]:
        """
        Auto-detect whether binary or text COLMAP files are present.

        Returns:
            (use_binary, cameras_file, images_file, points3d_file)
        """
        base = Path(colmap_path)

        # Prefer binary
        cameras_bin  = base / "cameras.bin"
        images_bin   = base / "images.bin"
        points3d_bin = base / "points3D.bin"
        if cameras_bin.exists() and images_bin.exists() and points3d_bin.exists():
            return True, str(cameras_bin), str(images_bin), str(points3d_bin)

        # Fall back to text
        cameras_txt  = base / "cameras.txt"
        images_txt   = base / "images.txt"
        points3d_txt = base / "points3D.txt"
        if cameras_txt.exists() and images_txt.exists() and points3d_txt.exists():
            return False, str(cameras_txt), str(images_txt), str(points3d_txt)

        raise FileNotFoundError(
            f"No COLMAP sparse files found in {colmap_path}. "
            "Expected cameras.{{bin,txt}}, images.{{bin,txt}}, points3D.{{bin,txt}}."
        )

    # ------------------------------------------------------------------
    # Public load method
    # ------------------------------------------------------------------

    def load(
        self,
        colmap_path: str,
        images_path: str,
        resolution: int = -1,
    ) -> Tuple[List[Camera], np.ndarray, np.ndarray]:
        """
        Load a COLMAP sparse reconstruction.

        Args:
            colmap_path: Path to COLMAP sparse reconstruction directory
                         (contains cameras.bin / cameras.txt, etc.).
            images_path: Path to the directory containing the raw images
                         (used to build image_path for each Camera).
            resolution:  If > 0, scale images so the longer edge equals
                         this value (in pixels); −1 keeps native resolution.

        Returns:
            cameras:  List[Camera] — one Camera per registered image,
                      sorted by image name for deterministic ordering.
            points3d: np.ndarray of shape (N, 3), float64 — world-space
                      point positions.
            colors:   np.ndarray of shape (N, 3), float32 — RGB colors
                      in [0, 1] for each point.
        """
        use_binary, cam_file, img_file, pts_file = self._locate_sparse(colmap_path)

        print(f"[ColmapSceneLoader] Loading {'binary' if use_binary else 'text'} "
              f"COLMAP data from: {colmap_path}")

        # --- Read raw COLMAP records ---------------------------------------
        if use_binary:
            raw_cameras  = _read_cameras_binary(cam_file)
            raw_images   = _read_images_binary(img_file)
            raw_points3d = _read_points3d_binary(pts_file)
        else:
            raw_cameras  = _read_cameras_text(cam_file)
            raw_images   = _read_images_text(img_file)
            raw_points3d = _read_points3d_text(pts_file)

        print(f"[ColmapSceneLoader]  Cameras: {len(raw_cameras)}, "
              f"Images: {len(raw_images)}, "
              f"Points3D: {len(raw_points3d)}")

        # --- Build intrinsics cache (one per camera model) ----------------
        intrinsics_cache: Dict[int, Tuple[CameraIntrinsics, int, int]] = {}
        for cam_id, rc in raw_cameras.items():
            intrinsics_cache[cam_id] = _make_intrinsics(rc, resolution)

        # --- Convert images to Camera objects -----------------------------
        images_path_obj = Path(images_path)
        cameras: List[Camera] = []

        # Sort by name for deterministic ordering across runs
        sorted_images = sorted(raw_images.values(), key=lambda im: im.name)

        for uid, rim in enumerate(sorted_images):
            if rim.camera_id not in intrinsics_cache:
                print(f"[ColmapSceneLoader] WARNING: image '{rim.name}' references "
                      f"unknown camera_id {rim.camera_id} — skipping.")
                continue

            intrinsics, final_w, final_h = intrinsics_cache[rim.camera_id]

            # Rotation matrix and translation (world → camera)
            R = _qvec2rotmat(rim.qvec)
            T = rim.tvec.copy()

            extrinsics = CameraExtrinsics(
                R=R.astype(np.float64),
                T=T.astype(np.float64),
            )

            # Resolve image path
            img_path = images_path_obj / rim.name
            image_path_str = str(img_path) if img_path.exists() else None

            cam = Camera(
                uid=uid,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                image_path=image_path_str,
            )
            cameras.append(cam)

        print(f"[ColmapSceneLoader] Built {len(cameras)} Camera objects "
              f"(resolution={'native' if resolution < 0 else resolution}).")

        # --- Assemble point cloud -----------------------------------------
        if raw_points3d:
            pts_list    = []
            colors_list = []
            for pt in raw_points3d.values():
                pts_list.append(pt.xyz)
                colors_list.append(pt.rgb)

            points3d_np = np.stack(pts_list,    axis=0).astype(np.float64)  # (N, 3)
            colors_np   = np.stack(colors_list, axis=0).astype(np.float32) / 255.0  # (N, 3) [0,1]
        else:
            print("[ColmapSceneLoader] WARNING: No 3D points found — returning empty point cloud.")
            points3d_np = np.zeros((0, 3), dtype=np.float64)
            colors_np   = np.zeros((0, 3), dtype=np.float32)

        print(f"[ColmapSceneLoader] Point cloud: {points3d_np.shape[0]} points.")
        return cameras, points3d_np, colors_np
