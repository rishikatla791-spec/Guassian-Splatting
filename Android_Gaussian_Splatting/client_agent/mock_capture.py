"""Mock Capture Dataset Generator for Android 3DGS Simulation.

Generates synthetic ARCore `transforms.json` datasets with valid camera intrinsics,
homogenous 4x4 camera-to-world pose matrices, realistic spherical orbits,
initial 3D feature points, and multi-view JPEG frames packaged into a ZIP archive.
"""

import json
import math
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


def generate_camera_orbit_poses(
    num_frames: int = 12,
    radius: float = 2.0,
    elevation_deg: float = 25.0,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> List[np.ndarray]:
    """Generates a sequence of 4x4 camera-to-world matrices along a circular orbit.

    The camera points towards `center` with OpenCV/OpenGL convention:
    +X Right, +Y Up, -Z Forward into the scene.
    """
    poses = []
    elev_rad = math.radians(elevation_deg)
    y_val = radius * math.sin(elev_rad)
    r_xz = radius * math.cos(elev_rad)

    cx, cy, cz = center

    for i in range(num_frames):
        theta = (2.0 * math.pi * i) / max(1, num_frames)
        # Position of camera
        px = cx + r_xz * math.cos(theta)
        py = cy + y_val
        pz = cz + r_xz * math.sin(theta)
        cam_pos = np.array([px, py, pz], dtype=np.float32)

        # Forward vector pointing from camera to center
        forward = np.array([cx - px, cy - py, cz - pz], dtype=np.float32)
        forward = forward / np.linalg.norm(forward)

        # World up vector
        up_world = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        # Right vector = normalize(forward x up_world)
        right = np.cross(forward, up_world)
        norm_r = np.linalg.norm(right)
        if norm_r < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            right = right / norm_r

        # Orthogonal camera up vector = right x forward
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)

        # In OpenGL/NeRF coordinate system:
        # Camera looks down -Z, so cam_z = -forward
        cam_z = -forward
        cam_x = right
        cam_y = up

        # 4x4 Camera-to-World (c2w) matrix
        c2w = np.eye(4, dtype=np.float32)
        c2w[0:3, 0] = cam_x
        c2w[0:3, 1] = cam_y
        c2w[0:3, 2] = cam_z
        c2w[0:3, 3] = cam_pos

        poses.append(c2w)

    return poses


def create_synthetic_frame_image(
    width: int,
    height: int,
    frame_idx: int,
    num_frames: int,
    color_scheme: str = "gradient",
) -> Image.Image:
    """Creates a synthetic multi-view image with identifiable features and geometry."""
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)

    # Base background gradient based on frame index
    hue_phase = frame_idx / max(1, num_frames)
    r_base = int(40 + 80 * math.sin(hue_phase * 2 * math.pi))
    g_base = int(60 + 80 * math.sin(hue_phase * 2 * math.pi + 2.0))
    b_base = int(120 + 80 * math.cos(hue_phase * 2 * math.pi))

    # Top-to-bottom shading
    for y in range(height):
        factor = 0.5 + 0.5 * (y / height)
        r = min(255, int(r_base * factor))
        g = min(255, int(g_base * factor))
        b = min(255, int(b_base * factor))
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Draw synthetic foreground object (circle with parallax offset)
    cx = width // 2 + int(width * 0.1 * math.cos(hue_phase * 2 * math.pi))
    cy = height // 2 + int(height * 0.05 * math.sin(hue_phase * 2 * math.pi))
    radius = min(width, height) // 6

    draw.ellipse(
        [(cx - radius, cy - radius), (cx + radius, cy + radius)],
        fill=(240, 200, 80),
        outline=(255, 255, 255),
        width=3,
    )

    # Secondary feature for corner tracking
    draw.rectangle(
        [(cx - radius // 3, cy - radius // 3), (cx + radius // 3, cy + radius // 3)],
        fill=(180, 50, 50),
    )

    return img


def generate_mock_dataset(
    output_path: Optional[str] = None,
    num_frames: int = 8,
    width: int = 640,
    height: int = 480,
    fx: float = 525.0,
    fy: float = 525.0,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    num_points: int = 150,
    as_zip: bool = True,
) -> str:
    """Generates a complete synthetic ARCore dataset with transforms.json and images.

    Args:
        output_path: Target zip or directory path. If None, uses a temporary path.
        num_frames: Number of camera views.
        width: Image pixel width.
        height: Image pixel height.
        fx: Focal length in X.
        fy: Focal length in Y.
        cx: Principal point X. Defaults to width / 2.
        cy: Principal point Y. Defaults to height / 2.
        num_points: Number of synthetic 3D feature points to include.
        as_zip: If True, packages as a .zip file.

    Returns:
        Absolute path to the created dataset (.zip or folder).
    """
    if cx is None:
        cx = width / 2.0
    if cy is None:
        cy = height / 2.0

    temp_dir = Path(tempfile.mkdtemp(prefix="mock_capture_"))
    images_dir = temp_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    poses = generate_camera_orbit_poses(num_frames=num_frames)
    frames_meta = []

    for i, c2w in enumerate(poses):
        fname = f"frame_{i:04d}.jpg"
        img_path = images_dir / fname
        img = create_synthetic_frame_image(width, height, i, num_frames)
        img.save(img_path, format="JPEG", quality=90)

        # Row-major 4x4 matrix representation
        matrix_list = c2w.tolist()
        frames_meta.append({
            "file_path": f"images/{fname}",
            "timestamp_ns": int(1700000000000000000 + i * 33333333),
            "sharpness_score": round(85.0 + 10.0 * math.sin(i), 2),
            "mean_luminance": round(120.0 + 5.0 * math.cos(i), 2),
            "transform_matrix": matrix_list,
        })

    fov_x = 2.0 * math.atan(width / (2.0 * fx))
    fov_y = 2.0 * math.atan(height / (2.0 * fy))

    transforms_json_data = {
        "camera_model": "OPENCV",
        "camera_angle_x": fov_x,
        "camera_angle_y": fov_y,
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "w": width,
        "h": height,
        "system_source": "ARCore_6DoF_SLAM",
        "frames": frames_meta,
    }

    transforms_file = temp_dir / "transforms.json"
    with open(transforms_file, "w", encoding="utf-8") as f:
        json.dump(transforms_json_data, f, indent=2)

    # Initial sparse 3D point cloud
    pts_list = []
    np.random.seed(42)
    for pt_id in range(num_points):
        # Sample points in a unit sphere
        pt_xyz = (np.random.rand(3) - 0.5) * 1.5
        confidence = float(0.5 + 0.5 * np.random.rand())
        pts_list.append({
            "id": pt_id,
            "xyz": pt_xyz.tolist(),
            "confidence": round(confidence, 3),
        })

    points_file = temp_dir / "points3D_initial.json"
    with open(points_file, "w", encoding="utf-8") as f:
        json.dump({"num_points": len(pts_list), "points": pts_list}, f, indent=2)

    if as_zip:
        if output_path is None:
            fd, final_path = tempfile.mkstemp(suffix=".zip", prefix="mock_scan_")
            os.close(fd)
        else:
            final_path = output_path
            Path(final_path).parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(final_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(transforms_file, arcname="transforms.json")
            zf.write(points_file, arcname="points3D_initial.json")
            for img_file in images_dir.glob("*.jpg"):
                zf.write(img_file, arcname=f"images/{img_file.name}")

        shutil.rmtree(temp_dir, ignore_errors=True)
        return str(Path(final_path).resolve())
    else:
        if output_path is not None:
            dest_dir = Path(output_path)
            shutil.copytree(temp_dir, dest_dir, dirs_exist_ok=True)
            shutil.rmtree(temp_dir, ignore_errors=True)
            return str(dest_dir.resolve())
        return str(temp_dir.resolve())


def create_corrupt_dataset(
    defect_type: str = "missing_transforms",
    output_zip: Optional[str] = None,
) -> str:
    """Generates defect datasets for negative and boundary testing.

    Defect types:
    - 'missing_transforms': ZIP containing images but no transforms.json
    - 'empty_frames': transforms.json with frames: []
    - 'invalid_json': Corrupted non-parseable JSON content
    - 'missing_intrinsics': transforms.json without fl_x, fl_y, w, h
    - 'singular_matrix': transforms.json with a matrix containing zeros / det=0
    - 'missing_images': transforms.json referencing non-existent image files
    - 'nan_coordinates': transforms.json containing NaN / Inf values
    - 'empty_zip': Completely empty 0-byte or 22-byte empty zip
    """
    if output_zip is None:
        fd, output_zip = tempfile.mkstemp(suffix=".zip", prefix=f"corrupt_{defect_type}_")
        os.close(fd)

    temp_dir = Path(tempfile.mkdtemp(prefix="corrupt_ds_"))
    images_dir = temp_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if defect_type == "empty_zip":
        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED):
            pass
        shutil.rmtree(temp_dir, ignore_errors=True)
        return output_zip

    # Make 1 image frame
    img = Image.new("RGB", (320, 240), color=(100, 100, 100))
    img.save(images_dir / "frame_0000.jpg", format="JPEG")

    if defect_type == "missing_transforms":
        pass  # Do not write transforms.json
    elif defect_type == "empty_frames":
        data = {"fl_x": 500.0, "fl_y": 500.0, "w": 320, "h": 240, "frames": []}
        with open(temp_dir / "transforms.json", "w") as f:
            json.dump(data, f)
    elif defect_type == "invalid_json":
        with open(temp_dir / "transforms.json", "w") as f:
            f.write("{corrupt_json: true, unterminated: [1, 2,")
    elif defect_type == "missing_intrinsics":
        data = {
            "frames": [{
                "file_path": "images/frame_0000.jpg",
                "transform_matrix": np.eye(4).tolist(),
            }]
        }
        with open(temp_dir / "transforms.json", "w") as f:
            json.dump(data, f)
    elif defect_type == "singular_matrix":
        data = {
            "fl_x": 500.0,
            "fl_y": 500.0,
            "w": 320,
            "h": 240,
            "frames": [{
                "file_path": "images/frame_0000.jpg",
                "transform_matrix": np.zeros((4, 4)).tolist(),  # det = 0
            }],
        }
        with open(temp_dir / "transforms.json", "w") as f:
            json.dump(data, f)
    elif defect_type == "missing_images":
        data = {
            "fl_x": 500.0,
            "fl_y": 500.0,
            "w": 320,
            "h": 240,
            "frames": [{
                "file_path": "images/nonexistent_image_12345.jpg",
                "transform_matrix": np.eye(4).tolist(),
            }],
        }
        with open(temp_dir / "transforms.json", "w") as f:
            json.dump(data, f)
    elif defect_type == "nan_coordinates":
        mat = np.eye(4)
        mat[0, 3] = float("nan")
        # In json nan converts to null or non-standard NaN
        with open(temp_dir / "transforms.json", "w") as f:
            f.write(json.dumps({
                "fl_x": 500.0,
                "fl_y": 500.0,
                "w": 320,
                "h": 240,
                "frames": [{
                    "file_path": "images/frame_0000.jpg",
                    "transform_matrix": [
                        [1.0, 0.0, 0.0, "NaN"],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                }],
            }))

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        if (temp_dir / "transforms.json").exists():
            zf.write(temp_dir / "transforms.json", arcname="transforms.json")
        if defect_type != "missing_images":
            for img_file in images_dir.glob("*.jpg"):
                zf.write(img_file, arcname=f"images/{img_file.name}")

    shutil.rmtree(temp_dir, ignore_errors=True)
    return output_zip


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic ARCore dataset")
    parser.add_argument("--output", "-o", default="mock_capture.zip", help="Output path")
    parser.add_argument("--frames", "-n", type=int, default=12, help="Number of frames")
    parser.add_argument("--width", type=int, default=640, help="Image width")
    parser.add_argument("--height", type=int, default=480, help="Image height")
    args = parser.parse_args()

    out_file = generate_mock_dataset(
        output_path=args.output,
        num_frames=args.frames,
        width=args.width,
        height=args.height,
    )
    print(f"Generated mock ARCore dataset at: {out_file}")
