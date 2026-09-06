"""Tier 1 & Tier 2 Tests: Dataset Ingestion Bridge (ARCore to COLMAP).

Covers:
- Feature 4: Dataset Ingestion Bridge (ARCore transforms.json to COLMAP sparse/0)
- Feature 9: Pipeline Resilience & Ingestion Error Handling

Thresholds: >=5 tests per feature for Tier 1 (Happy Path) and Tier 2 (Boundary/Adversarial).
"""

import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from plyfile import PlyData
import pytest

# Ensure server module is discoverable
SERVER_DIR = Path(__file__).resolve().parent.parent.parent / "Android_Gaussian_Splatting" / "server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

CLIENT_AGENT_DIR = Path(__file__).resolve().parent.parent.parent / "Android_Gaussian_Splatting" / "client_agent"
if str(CLIENT_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(CLIENT_AGENT_DIR))

from dataset_adapter import (
    convert_transforms_to_colmap,
    convert_pose_opengl_to_colmap,
    rotmat2qvec,
    qvec2rotmat,
    generate_sparse_points,
    CorruptTransformsError,
    InvalidMatrixError,
    MissingIntrinsicsError,
)
from mock_capture import generate_mock_dataset, create_corrupt_dataset


@pytest.fixture
def mock_dataset_dir():
    """Generates an unzipped mock dataset folder and cleans up after test."""
    ds_dir = generate_mock_dataset(num_frames=6, width=640, height=480, as_zip=False)
    yield Path(ds_dir)
    shutil.rmtree(ds_dir, ignore_errors=True)


# =============================================================================
# FEATURE 4: INGESTION ADAPTER & FORMAT CONVERSION (TIER 1)
# =============================================================================

def test_convert_standard_mock_dataset(mock_dataset_dir):
    """T1-F4-01: Full conversion of standard ARCore dataset creates all sparse/0 files."""
    summary = convert_transforms_to_colmap(mock_dataset_dir)
    assert summary["status"] == "success"
    assert summary["num_images"] == 6
    assert summary["rejected_images"] == 0
    assert summary["num_points"] > 0

    sparse_dir = mock_dataset_dir / "sparse" / "0"
    assert (sparse_dir / "cameras.txt").exists()
    assert (sparse_dir / "images.txt").exists()
    assert (sparse_dir / "points3D.txt").exists()
    assert (sparse_dir / "points3D.ply").exists()


def test_colmap_cameras_format(mock_dataset_dir):
    """T1-F4-02: cameras.txt adheres strictly to COLMAP PINHOLE camera specification."""
    convert_transforms_to_colmap(mock_dataset_dir)
    cameras_file = mock_dataset_dir / "sparse" / "0" / "cameras.txt"

    lines = [line.strip() for line in cameras_file.read_text().splitlines() if line.strip() and not line.startswith("#")]
    assert len(lines) == 1

    parts = lines[0].split()
    assert parts[0] == "1"  # CAMERA_ID
    assert parts[1] == "PINHOLE"  # MODEL
    w, h = int(parts[2]), int(parts[3])
    fx, fy, cx, cy = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
    assert w == 640
    assert h == 480
    assert fx > 0 and fy > 0
    assert cx > 0 and cy > 0


def test_colmap_images_format(mock_dataset_dir):
    """T1-F4-03: images.txt adheres strictly to COLMAP image and pose specifications."""
    convert_transforms_to_colmap(mock_dataset_dir)
    images_file = mock_dataset_dir / "sparse" / "0" / "images.txt"

    lines = [line.strip() for line in images_file.read_text().splitlines() if line.strip() and not line.startswith("#")]
    # COLMAP format has 2 lines per image (image pose line, followed by empty points2D line)
    # Since we filter empty lines, lines will contain 1 line per image
    assert len(lines) == 6

    for idx, line in enumerate(lines):
        parts = line.split()
        img_id = int(parts[0])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        cam_id = int(parts[8])
        name = parts[9]

        assert img_id == idx + 1
        assert cam_id == 1
        assert name == f"frame_{idx:04d}.jpg"

        # Quaternion normalization check: qw^2 + qx^2 + qy^2 + qz^2 ≈ 1.0
        q_norm = math.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        assert abs(q_norm - 1.0) < 1e-4

        # Translation vector finite check
        assert all(math.isfinite(t) for t in (tx, ty, tz))


def test_coordinate_frame_transformation():
    """T1-F4-04: Mathematically verifies OpenGL camera-to-world to COLMAP world-to-camera."""
    # Construct an identity camera-to-world (camera at origin looking down -Z, +Y up)
    c2w = np.eye(4, dtype=np.float64)

    R_w2c, T_w2c, qvec = convert_pose_opengl_to_colmap(c2w, invert_yz=True)

    # Inverting Y and Z:
    # c2w_adj column 1 is [0, -1, 0], column 2 is [0, 0, -1]
    # w2c = inv(c2w_adj)
    # R_w2c = diag([1, -1, -1])
    expected_R = np.diag([1.0, -1.0, -1.0])
    assert np.allclose(R_w2c, expected_R, atol=1e-5)

    # Translation for origin camera should be [0, 0, 0]
    assert np.allclose(T_w2c, np.zeros(3), atol=1e-5)

    # Quaternion norm must be 1.0
    assert abs(np.linalg.norm(qvec) - 1.0) < 1e-5


def test_sparse_point_cloud_generation(mock_dataset_dir):
    """T1-F4-05: Verifies points3D.ply and points3D.txt generation and bounds."""
    convert_transforms_to_colmap(mock_dataset_dir, num_synthetic_points=500)
    ply_path = mock_dataset_dir / "sparse" / "0" / "points3D.ply"
    txt_path = mock_dataset_dir / "sparse" / "0" / "points3D.txt"

    assert ply_path.exists()
    assert txt_path.exists()

    plydata = PlyData.read(str(ply_path))
    v = plydata["vertex"]
    assert len(v) >= 100

    # Ensure all coordinates are finite
    xs, ys, zs = v["x"], v["y"], v["z"]
    assert np.all(np.isfinite(xs))
    assert np.all(np.isfinite(ys))
    assert np.all(np.isfinite(zs))

    # Ensure RGB colors are valid uint8 [0, 255]
    rs, gs, bs = v["red"], v["green"], v["blue"]
    assert np.all(rs >= 0) and np.all(rs <= 255)
    assert np.all(gs >= 0) and np.all(gs <= 255)
    assert np.all(bs >= 0) and np.all(bs <= 255)


def test_initial_points_json_ingestion(mock_dataset_dir):
    """T1-F4-06: Verifies points3D_initial.json is favored when present."""
    pts_init = mock_dataset_dir / "points3D_initial.json"
    assert pts_init.exists()

    summary = convert_transforms_to_colmap(mock_dataset_dir)
    assert summary["status"] == "success"
    # The mock dataset creates 150 points in points3D_initial.json
    assert summary["num_points"] == 150


def test_quaternion_matrix_roundtrip():
    """T1-F4-07: Verifies rotmat2qvec and qvec2rotmat bidirectional consistency."""
    # Test random SO(3) rotations
    np.random.seed(123)
    for _ in range(10):
        # Generate random orthogonal matrix via QR decomposition
        H = np.random.randn(3, 3)
        Q, R = np.linalg.qr(H)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1

        qvec = rotmat2qvec(Q)
        assert abs(np.linalg.norm(qvec) - 1.0) < 1e-6
        assert qvec[0] >= 0  # Canonical non-negative qw

        Q_recon = qvec2rotmat(qvec)
        assert np.allclose(Q, Q_recon, atol=1e-5)


# =============================================================================
# FEATURE 4 & 9: BOUNDARY, CORNER & NEGATIVE ADAPTER CASES (TIER 2)
# =============================================================================

def test_adapter_missing_transforms_json():
    """T2-F4-01: Directory lacking transforms.json raises CorruptTransformsError."""
    temp_empty_dir = Path(tempfile.mkdtemp(prefix="empty_scan_"))
    try:
        with pytest.raises(CorruptTransformsError, match="transforms.json not found"):
            convert_transforms_to_colmap(temp_empty_dir)
    finally:
        shutil.rmtree(temp_empty_dir, ignore_errors=True)


def test_adapter_empty_frames_list():
    """T2-F4-02: transforms.json with empty frames list raises CorruptTransformsError."""
    corrupt_zip = create_corrupt_dataset("empty_frames")
    extract_dir = Path(tempfile.mkdtemp(prefix="test_empty_frames_"))
    try:
        import zipfile
        with zipfile.ZipFile(corrupt_zip, "r") as zf:
            zf.extractall(extract_dir)

        with pytest.raises(CorruptTransformsError, match="No frames found"):
            convert_transforms_to_colmap(extract_dir)
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)
        if os.path.exists(corrupt_zip):
            os.unlink(corrupt_zip)


def test_adapter_malformed_json():
    """T2-F4-03: Corrupt non-parseable JSON raises CorruptTransformsError."""
    corrupt_zip = create_corrupt_dataset("invalid_json")
    extract_dir = Path(tempfile.mkdtemp(prefix="test_invalid_json_"))
    try:
        import zipfile
        with zipfile.ZipFile(corrupt_zip, "r") as zf:
            zf.extractall(extract_dir)

        with pytest.raises(CorruptTransformsError, match="Malformed JSON"):
            convert_transforms_to_colmap(extract_dir)
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)
        if os.path.exists(corrupt_zip):
            os.unlink(corrupt_zip)


def test_adapter_missing_intrinsics_fallback(mock_dataset_dir):
    """T2-F4-04: Resolves intrinsics when fl_x/fl_y are omitted but camera_angle_x exists."""
    transforms_file = mock_dataset_dir / "transforms.json"
    with open(transforms_file, "r") as f:
        data = json.load(f)

    # Delete explicit focal lengths
    del data["fl_x"]
    del data["fl_y"]
    data["camera_angle_x"] = 1.0472  # 60 degrees

    with open(transforms_file, "w") as f:
        json.dump(data, f)

    summary = convert_transforms_to_colmap(mock_dataset_dir)
    assert summary["status"] == "success"
    assert summary["intrinsics"]["fx"] > 0


def test_adapter_singular_transform_matrix(mock_dataset_dir):
    """T2-F4-05: Non-invertible transform matrix in a frame is skipped gracefully."""
    transforms_file = mock_dataset_dir / "transforms.json"
    with open(transforms_file, "r") as f:
        data = json.load(f)

    # Insert a singular frame (all zeros, det=0)
    singular_frame = {
        "file_path": "images/frame_singular.jpg",
        "transform_matrix": [[0.0] * 4 for _ in range(4)],
    }
    data["frames"].append(singular_frame)

    with open(transforms_file, "w") as f:
        json.dump(data, f)

    summary = convert_transforms_to_colmap(mock_dataset_dir)
    assert summary["status"] == "success"
    assert summary["rejected_images"] == 1
    assert summary["num_images"] == 6

    # If ALL frames are singular, it must raise InvalidMatrixError
    for frame in data["frames"]:
        frame["transform_matrix"] = [[0.0] * 4 for _ in range(4)]

    with open(transforms_file, "w") as f:
        json.dump(data, f)

    with pytest.raises(InvalidMatrixError):
        convert_transforms_to_colmap(mock_dataset_dir)


def test_adapter_nan_coordinates_rejected(mock_dataset_dir):
    """T2-F4-06: Transform matrix with NaN/Inf values is rejected."""
    transforms_file = mock_dataset_dir / "transforms.json"
    with open(transforms_file, "r") as f:
        data = json.load(f)

    # Set all frames to have NaN coordinates
    for frame in data["frames"]:
        mat = np.eye(4).tolist()
        mat[0][3] = float("nan")
        frame["transform_matrix"] = mat

    with open(transforms_file, "w") as f:
        # Write JSON with null/nan
        f.write(json.dumps(data))

    with pytest.raises(InvalidMatrixError):
        convert_transforms_to_colmap(mock_dataset_dir)


def test_adapter_non_existent_scan_dir():
    """T2-F4-07: Passing non-existent directory raises CorruptTransformsError."""
    with pytest.raises(CorruptTransformsError, match="does not exist"):
        convert_transforms_to_colmap("C:/non_existent_path_xyz_123456")


def test_adapter_custom_sparse_output_dir(mock_dataset_dir):
    """T2-F4-08: Explicit output_sparse_dir directs artifacts to the requested path."""
    custom_sparse = Path(tempfile.mkdtemp(prefix="custom_sparse_"))
    try:
        summary = convert_transforms_to_colmap(
            mock_dataset_dir,
            output_sparse_dir=custom_sparse,
        )
        assert summary["status"] == "success"
        assert Path(summary["sparse_dir"]) == custom_sparse
        assert (custom_sparse / "cameras.txt").exists()
        assert (custom_sparse / "images.txt").exists()
        assert (custom_sparse / "points3D.ply").exists()
    finally:
        shutil.rmtree(custom_sparse, ignore_errors=True)
