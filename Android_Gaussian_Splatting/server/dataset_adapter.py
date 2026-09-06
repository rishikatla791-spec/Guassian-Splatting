"""
Dataset Adapter: Ingestion bridge converting ARCore/Nerfstudio transforms.json
into standard COLMAP sparse/0 format for Inria 3D Gaussian Splatting.

Converts:
  - ARCore / Nerfstudio camera poses (camera-to-world, OpenGL convention)
    into COLMAP world-to-camera format (OpenCV convention: X-right, Y-down, Z-forward).
  - Generates cameras.txt, images.txt, points3D.txt, and points3D.ply in sparse/0/.
  - Ingests points3D_initial.json or synthesizes a reliable initial 3D point cloud
    so Gaussian Splatting initialization initializes without crashing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import struct
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from plyfile import PlyData, PlyElement


class DatasetAdapterError(Exception):
    """Base exception for dataset adapter errors."""
    pass


class CorruptTransformsError(DatasetAdapterError):
    """Raised when transforms.json is missing, malformed, or unparseable."""
    pass


class InvalidMatrixError(DatasetAdapterError):
    """Raised when a camera transform matrix is singular, degenerate, or has invalid dimensions."""
    pass


class MissingIntrinsicsError(DatasetAdapterError):
    """Raised when camera intrinsics cannot be resolved from transforms.json or images."""
    pass


def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to a normalized quaternion [qw, qx, qy, qz].
    COLMAP convention uses [qw, qx, qy, qz].
    """
    try:
        from scipy.spatial.transform import Rotation
        r = Rotation.from_matrix(R)
        # scipy returns [qx, qy, qz, qw]
        q_scipy = r.as_quat()
        qvec = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]], dtype=np.float64)
        if qvec[0] < 0:
            qvec = -qvec
        return qvec
    except Exception:
        # Numerically stable pure-numpy fallback (Shepperd's algorithm)
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2.0
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
        qvec = np.array([qw, qx, qy, qz], dtype=np.float64)
        if qvec[0] < 0:
            qvec = -qvec
        norm = np.linalg.norm(qvec)
        return qvec / (norm if norm > 1e-12 else 1.0)



def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert a quaternion [qw, qx, qy, qz] to a 3x3 rotation matrix."""
    qw, qx, qy, qz = qvec
    return np.array([
        [1.0 - 2.0 * qy**2 - 2.0 * qz**2,
         2.0 * qx * qy - 2.0 * qw * qz,
         2.0 * qx * qz + 2.0 * qw * qy],
        [2.0 * qx * qy + 2.0 * qw * qz,
         1.0 - 2.0 * qx**2 - 2.0 * qz**2,
         2.0 * qy * qz - 2.0 * qw * qx],
        [2.0 * qx * qz - 2.0 * qw * qy,
         2.0 * qy * qz + 2.0 * qw * qx,
         1.0 - 2.0 * qx**2 - 2.0 * qy**2]
    ], dtype=np.float64)


def parse_matrix_from_frame(frame: Dict[str, Any], frame_idx: int = 0) -> np.ndarray:
    """
    Extracts a 4x4 camera-to-world transformation matrix from a frame dict.
    Supports:
      - 'transform_matrix': list of 4 lists of 4 numbers
      - 'rotation' (quaternion or 3x3 matrix) + 'position'/'translation'
    """
    if "transform_matrix" in frame:
        mat_data = frame["transform_matrix"]
        try:
            mat = np.array(mat_data, dtype=np.float64)
        except Exception as e:
            raise InvalidMatrixError(f"Frame {frame_idx}: unable to parse transform_matrix: {e}")
        if mat.shape != (4, 4):
            raise InvalidMatrixError(f"Frame {frame_idx}: transform_matrix shape is {mat.shape}, expected (4, 4)")
        if not np.all(np.isfinite(mat)):
            raise InvalidMatrixError(f"Frame {frame_idx}: transform_matrix contains NaN or Inf values")
        return mat

    # Fallback to separate rotation and translation
    pos = frame.get("position") or frame.get("translation")
    rot = frame.get("rotation") or frame.get("quaternion")
    if pos is not None and rot is not None:
        try:
            pos_arr = np.array(pos, dtype=np.float64).reshape(3)
            rot_arr = np.array(rot, dtype=np.float64)
            mat = np.eye(4, dtype=np.float64)
            mat[:3, 3] = pos_arr
            if rot_arr.shape == (3, 3):
                mat[:3, :3] = rot_arr
            elif rot_arr.size == 4:
                # [qw, qx, qy, qz] or [qx, qy, qz, qw]
                # If scalar is first or last: detect norm
                q = rot_arr.flatten()
                if abs(np.linalg.norm(q) - 1.0) > 0.2:
                    q = q / (np.linalg.norm(q) + 1e-8)
                # Convention: frame might have [qw, qx, qy, qz]
                mat[:3, :3] = qvec2rotmat(q)
            else:
                raise InvalidMatrixError(f"Frame {frame_idx}: unexpected rotation format shape {rot_arr.shape}")
            if not np.all(np.isfinite(mat)):
                raise InvalidMatrixError(f"Frame {frame_idx}: matrix contains NaN/Inf")
            return mat
        except Exception as e:
            raise InvalidMatrixError(f"Frame {frame_idx}: failed to construct matrix from pose: {e}")

    raise InvalidMatrixError(f"Frame {frame_idx}: missing transform_matrix or rotation/position keys")


def convert_pose_opengl_to_colmap(
    c2w: np.ndarray,
    invert_yz: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a camera-to-world matrix (OpenGL: X right, Y up, Z back)
    to COLMAP world-to-camera parameters (OpenCV: X right, Y down, Z forward).

    Args:
        c2w: 4x4 camera-to-world matrix
        invert_yz: If True, flips Y and Z camera axes from OpenGL to OpenCV/COLMAP convention.

    Returns:
        (R_w2c, T_w2c, qvec)
        - R_w2c: 3x3 rotation matrix (world to camera)
        - T_w2c: 3-element translation vector (world to camera)
        - qvec: [qw, qx, qy, qz] quaternion
    """
    if c2w.shape != (4, 4):
        raise InvalidMatrixError(f"c2w matrix must be 4x4, got {c2w.shape}")

    c2w_adj = c2w.copy()
    if invert_yz:
        # Multiply columns 1 and 2 (Y and Z) by -1
        c2w_adj[:3, 1:3] *= -1.0

    # Invert camera-to-world to obtain world-to-camera
    try:
        w2c = np.linalg.inv(c2w_adj)
    except np.linalg.LinAlgError as e:
        raise InvalidMatrixError(f"Failed to invert transform matrix (singular): {e}")

    R_w2c = w2c[:3, :3]
    T_w2c = w2c[:3, 3]
    qvec = rotmat2qvec(R_w2c)
    return R_w2c, T_w2c, qvec


def generate_sparse_points(
    c2w_matrices: List[np.ndarray],
    initial_points_file: Optional[Path] = None,
    num_synthetic_points: int = 1500
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads initial 3D sparse points from points3D_initial.json if present,
    or synthesizes a high-quality initial point cloud based on camera centers
    and viewing directions.

    Returns:
        (points, colors)
        - points: np.ndarray of shape (N, 3), float32
        - colors: np.ndarray of shape (N, 3), uint8 in [0, 255]
    """
    if initial_points_file and initial_points_file.exists():
        try:
            with open(initial_points_file, "r", encoding="utf-8") as f:
                pt_data = json.load(f)
            pts_list = []
            cols_list = []
            items = pt_data.get("points", [])
            for item in items:
                xyz = item.get("xyz")
                if xyz and len(xyz) == 3 and all(np.isfinite(xyz)):
                    pts_list.append(xyz)
                    # RGB color if present
                    rgb = item.get("rgb", [180, 180, 180])
                    cols_list.append(rgb[:3])

            if len(pts_list) >= 50:
                pts_arr = np.array(pts_list, dtype=np.float32)
                cols_arr = np.clip(np.array(cols_list, dtype=np.float32), 0, 255).astype(np.uint8)
                return pts_arr, cols_arr
        except Exception:
            pass  # Fall through to synthetic generation

    # Synthetic initialization from camera distribution
    if not c2w_matrices:
        # Minimal default cloud
        pts = np.random.uniform(-1.0, 1.0, size=(num_synthetic_points, 3)).astype(np.float32)
        cols = np.full((num_synthetic_points, 3), 180, dtype=np.uint8)
        return pts, cols

    cam_centers = np.array([m[:3, 3] for m in c2w_matrices], dtype=np.float32)
    center = np.median(cam_centers, axis=0)
    dists = np.linalg.norm(cam_centers - center, axis=1)
    radius = float(np.percentile(dists, 90)) if len(dists) > 0 else 1.0
    radius = max(radius, 0.2)

    # Compute scene focus point by looking forward along camera viewing vectors
    # Camera forward in OpenGL is -Z (column 2 negated)
    look_points = []
    for m in c2w_matrices:
        forward = -m[:3, 2]  # OpenGL forward direction
        forward_norm = np.linalg.norm(forward)
        if forward_norm > 1e-6:
            forward = forward / forward_norm
            look_points.append(m[:3, 3] + forward * (radius * 1.2))

    if look_points:
        scene_focus = np.median(look_points, axis=0)
    else:
        scene_focus = center

    # Generate points around the focus and bounded sphere
    np.random.seed(42)
    # Core cluster near focal region
    n_core = int(num_synthetic_points * 0.7)
    core_pts = scene_focus + np.random.normal(0.0, radius * 0.4, size=(n_core, 3))

    # Surrounding shell
    n_shell = num_synthetic_points - n_core
    shell_dir = np.random.normal(0.0, 1.0, size=(n_shell, 3))
    shell_dir /= (np.linalg.norm(shell_dir, axis=1, keepdims=True) + 1e-8)
    shell_rad = np.random.uniform(radius * 0.2, radius * 1.5, size=(n_shell, 1))
    shell_pts = scene_focus + shell_dir * shell_rad

    points = np.vstack([core_pts, shell_pts]).astype(np.float32)
    # Give pleasant warm neutral colors
    colors = np.full((num_synthetic_points, 3), 170, dtype=np.uint8)
    colors[:, 0] = np.random.randint(150, 200, size=num_synthetic_points)
    colors[:, 1] = np.random.randint(150, 195, size=num_synthetic_points)
    colors[:, 2] = np.random.randint(140, 190, size=num_synthetic_points)

    return points, colors


def write_cameras_txt(
    path: Path,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    camera_id: int = 1
) -> None:
    """
    Write COLMAP cameras.txt using standard PINHOLE model:
    CAMERA_ID PINHOLE WIDTH HEIGHT fx fy cx cy
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(f"{camera_id} PINHOLE {width} {height} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")


def write_images_txt(
    path: Path,
    image_records: List[Dict[str, Any]]
) -> None:
    """
    Write COLMAP images.txt format.
    Two lines per registered image:
      Line 1: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
      Line 2: POINTS2D[] (can be empty)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(image_records)}\n")
        for rec in image_records:
            qw, qx, qy, qz = rec["qvec"]
            tx, ty, tz = rec["tvec"]
            img_id = rec["image_id"]
            cam_id = rec["camera_id"]
            name = rec["name"]
            f.write(f"{img_id} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {tx:.9f} {ty:.9f} {tz:.9f} {cam_id} {name}\n")
            f.write("\n")  # Empty line for points2D


def write_points3D_txt(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray
) -> None:
    """
    Write COLMAP points3D.txt format:
    POINT3D_ID X Y Z R G B ERROR TRACK[]
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points)}\n")
        for i in range(len(points)):
            x, y, z = points[i]
            r, g, b = colors[i]
            # ID starts at 1
            f.write(f"{i + 1} {x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)} 1.0\n")


def write_points3D_ply(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray
) -> None:
    """
    Write standard PLY file with x, y, z, nx, ny, nz, red, green, blue.
    Inria's dataset_readers fetchPly directly reads this format.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ]
    num_pts = len(points)
    elements = np.empty(num_pts, dtype=dtype)
    elements['x'] = points[:, 0]
    elements['y'] = points[:, 1]
    elements['z'] = points[:, 2]
    elements['nx'] = 0.0
    elements['ny'] = 0.0
    elements['nz'] = 0.0
    elements['red'] = colors[:, 0]
    elements['green'] = colors[:, 1]
    elements['blue'] = colors[:, 2]

    vertex_element = PlyElement.describe(elements, 'vertex')
    PlyData([vertex_element]).write(str(path))


def convert_transforms_to_colmap(
    scan_dir: Union[str, Path],
    output_sparse_dir: Optional[Union[str, Path]] = None,
    opengl_to_colmap: bool = True,
    num_synthetic_points: int = 1500
) -> Dict[str, Any]:
    """
    Main ingestion bridge entrypoint.
    Converts ARCore / Nerfstudio `transforms.json` into standard COLMAP `sparse/0` format.

    Args:
        scan_dir: Directory containing `transforms.json` and image files/folders.
        output_sparse_dir: Output directory for sparse/0/. Defaults to `<scan_dir>/sparse/0`.
        opengl_to_colmap: If True, applies camera frame Y/Z inversion from OpenGL to COLMAP.
        num_synthetic_points: Number of sparse points to generate if points3D_initial.json is absent.

    Returns:
        Summary dict containing conversion statistics.
    """
    scan_dir = Path(scan_dir).resolve()
    if not scan_dir.exists():
        raise CorruptTransformsError(f"Scan directory does not exist: {scan_dir}")

    # Locate transforms.json
    transforms_file = scan_dir / "transforms.json"
    if not transforms_file.exists():
        # Check alternative names
        alt_files = list(scan_dir.glob("*transforms*.json"))
        if alt_files:
            transforms_file = alt_files[0]
        else:
            raise CorruptTransformsError(f"transforms.json not found in {scan_dir}")

    try:
        with open(transforms_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise CorruptTransformsError(f"Malformed JSON in {transforms_file}: {e}")
    except Exception as e:
        raise CorruptTransformsError(f"Failed to read {transforms_file}: {e}")

    if not isinstance(data, dict):
        raise CorruptTransformsError(f"Invalid JSON root in {transforms_file}: expected object")

    frames = data.get("frames", [])
    if not isinstance(frames, list) or len(frames) == 0:
        raise CorruptTransformsError(f"No frames found in {transforms_file}")

    # Camera Intrinsics
    w = data.get("w")
    h = data.get("h")
    fx = data.get("fl_x")
    fy = data.get("fl_y")
    cx = data.get("cx")
    cy = data.get("cy")

    # If focal length is missing, attempt from camera_angle_x
    if (fx is None or fy is None) and "camera_angle_x" in data:
        angle_x = float(data["camera_angle_x"])
        if w is None:
            w = 1280  # temporary fallback for fov calculation
        fx = float(w) / (2.0 * math.tan(angle_x / 2.0))
        fy = fx

    # If width/height missing, attempt to inspect the first image file
    first_frame_path = frames[0].get("file_path", "")
    resolved_img_path = None
    if first_frame_path:
        for candidate in [
            scan_dir / first_frame_path,
            scan_dir / "images" / Path(first_frame_path).name,
            scan_dir / Path(first_frame_path).name
        ]:
            if candidate.exists():
                resolved_img_path = candidate
                break

    if resolved_img_path and (w is None or h is None):
        try:
            from PIL import Image as PILImage
            with PILImage.open(resolved_img_path) as img:
                w, h = img.size
        except Exception:
            pass

    # Defaults if still unassigned
    w = int(w or 1280)
    h = int(h or 720)
    fx = float(fx or (w * 0.8))
    fy = float(fy or fx)
    cx = float(cx or (w / 2.0))
    cy = float(cy or (h / 2.0))

    if fx <= 0 or fy <= 0 or w <= 0 or h <= 0:
        raise MissingIntrinsicsError(f"Invalid camera intrinsics: w={w}, h={h}, fx={fx}, fy={fy}")

    # Process Frames
    image_records = []
    c2w_matrices = []
    rejected_frames = 0

    for idx, frame in enumerate(frames):
        try:
            c2w = parse_matrix_from_frame(frame, frame_idx=idx)
            R_w2c, T_w2c, qvec = convert_pose_opengl_to_colmap(c2w, invert_yz=opengl_to_colmap)
        except InvalidMatrixError:
            rejected_frames += 1
            continue

        c2w_matrices.append(c2w)

        # Determine image file name
        raw_path = frame.get("file_path", f"frame_{idx:05d}.jpg")
        image_name = Path(raw_path).name




        image_records.append({
            "image_id": len(image_records) + 1,
            "qvec": qvec,
            "tvec": T_w2c,
            "camera_id": 1,
            "name": image_name,
            "c2w": c2w
        })

    if len(image_records) == 0:
        raise InvalidMatrixError("No valid camera frames could be parsed from transforms.json")

    # Output paths
    if output_sparse_dir is None:
        sparse_dir = scan_dir / "sparse" / "0"
    else:
        sparse_dir = Path(output_sparse_dir).resolve()
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # Initial Point Cloud
    points_init_file = scan_dir / "points3D_initial.json"
    points, colors = generate_sparse_points(
        c2w_matrices,
        initial_points_file=points_init_file,
        num_synthetic_points=num_synthetic_points
    )

    # Write COLMAP sparse reconstruction files
    write_cameras_txt(sparse_dir / "cameras.txt", width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy)
    write_images_txt(sparse_dir / "images.txt", image_records)
    write_points3D_txt(sparse_dir / "points3D.txt", points, colors)
    write_points3D_ply(sparse_dir / "points3D.ply", points, colors)

    summary = {
        "status": "success",
        "scan_dir": str(scan_dir),
        "sparse_dir": str(sparse_dir),
        "num_cameras": 1,
        "num_images": len(image_records),
        "rejected_images": rejected_frames,
        "num_points": len(points),
        "intrinsics": {
            "width": w,
            "height": h,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy
        }
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Convert ARCore transforms.json to COLMAP sparse/0 format.")
    parser.add_argument("scan_dir", type=str, help="Directory containing transforms.json and images")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output sparse/0 directory (default: <scan_dir>/sparse/0)")
    parser.add_argument("--no-yz-flip", action="store_true", help="Do not invert Y and Z axes (for non-OpenGL datasets)")
    parser.add_argument("--num-points", type=int, default=1500, help="Initial synthetic sparse points count")

    args = parser.parse_args()
    try:
        summary = convert_transforms_to_colmap(
            scan_dir=args.scan_dir,
            output_sparse_dir=args.output,
            opengl_to_colmap=not args.no_yz_flip,
            num_synthetic_points=args.num_points
        )
        print(json.dumps(summary, indent=2))
        sys.exit(0)
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
