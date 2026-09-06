"""
Unit tests for Production ARCore Dataset Validator & Loader.
"""

import json
import tempfile
from pathlib import Path
import numpy as np
import pytest
from PIL import Image as PILImage

from gaussian.pipeline.arcore_dataset_validator import ARCoreDatasetValidator


@pytest.fixture
def mock_arcore_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        img_dir = tmp_path / "images"
        img_dir.mkdir(parents=True)

        # Create dummy frame image
        img = PILImage.new("RGB", (640, 480), color=(128, 128, 128))
        img.save(img_dir / "frame_0000.jpg")

        # Identity 4x4 matrix
        mat = np.eye(4, dtype=np.float64).tolist()

        transforms = {
            "w": 640,
            "h": 480,
            "fl_x": 500.0,
            "fl_y": 500.0,
            "cx": 320.0,
            "cy": 240.0,
            "frames": [
                {
                    "file_path": "images/frame_0000.jpg",
                    "timestamp_ns": 1000000,
                    "transform_matrix": mat,
                    "sharpness_score": 150.0,
                    "mean_luminance": 120.0,
                }
            ]
        }

        with open(tmp_path / "transforms.json", "w") as f:
            json.dump(transforms, f)

        # Mock initial point cloud
        points = {
            "num_points": 2,
            "points": [
                {"id": 0, "xyz": [0.0, 0.0, 1.0], "confidence": 0.9},
                {"id": 1, "xyz": [0.5, 0.5, 2.0], "confidence": 0.8},
            ]
        }
        with open(tmp_path / "points3D_initial.json", "w") as f:
            json.dump(points, f)

        yield tmp_path


def test_arcore_validator_success(mock_arcore_dataset):
    validator = ARCoreDatasetValidator(mock_arcore_dataset)
    report, data = validator.validate()

    assert report.is_valid is True
    assert report.passed_frames == 1
    assert report.rejected_frames == 0
    assert report.num_points == 2


def test_arcore_validator_load_cameras(mock_arcore_dataset):
    validator = ARCoreDatasetValidator(mock_arcore_dataset)
    cameras = validator.load_as_cameras()

    assert len(cameras) == 1
    assert cameras[0].width == 640
    assert cameras[0].height == 480


def test_arcore_validator_point_cloud_load(mock_arcore_dataset):
    validator = ARCoreDatasetValidator(mock_arcore_dataset)
    pts, colors = validator.load_initial_point_cloud()

    assert pts.shape == (2, 3)
    assert colors.shape == (2, 3)
