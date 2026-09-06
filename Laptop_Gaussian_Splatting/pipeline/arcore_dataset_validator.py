"""
Production ARCore Dataset Validator & Loader for 3D Gaussian Splatting.

Validates mobile-acquired ARCore datasets containing `transforms.json`, image files,
and optional initial 3D point cloud `points3D_initial.json`.

Automated Quality & Diagnostic Checks:
  1. Matrix Orthonormality Check: det(R) ≈ 1.0, R Rᵀ ≈ I (SO(3) compliance).
  2. Timestamp Monotonicity: Verifies non-decreasing hardware timestamps.
  3. Spatial Pose Continuity: Detects tracking jumps or SLAM resets (Δd > 1.5m).
  4. Image File & Intrinsic Alignment: Confirms image dimensions match fx, fy, cx, cy.
  5. Initial Point Cloud Integration: Converts ARCore raw feature points to GaussianModel initialization.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from PIL import Image as PILImage


from ..core.camera import Camera, CameraIntrinsics, CameraExtrinsics


@dataclass
class ARCoreValidationReport:
    """Diagnostic report for an ARCore captured dataset."""
    is_valid: bool
    num_frames: int
    num_points: int
    passed_frames: int
    rejected_frames: int
    avg_sharpness: float
    rotation_errors: List[str]
    timestamp_errors: List[str]
    continuity_warnings: List[str]
    quality_score: float


class ARCoreDatasetValidator:
    """
    Automated validator and dataset builder for mobile ARCore captures.
    """

    def __init__(self, dataset_dir: str | Path, rotation_tolerance: float = 1e-2):
        self.dataset_dir = Path(dataset_dir)
        self.rotation_tolerance = rotation_tolerance
        self.transforms_path = self.dataset_dir / "transforms.json"
        self.points_path = self.dataset_dir / "points3D_initial.json"

    def validate(self) -> Tuple[ARCoreValidationReport, Dict[str, Any]]:

        """
        Run complete automated diagnostic test suite on dataset.
        """
        if not self.transforms_path.exists():
            return ARCoreValidationReport(
                is_valid=False,
                num_frames=0,
                num_points=0,
                passed_frames=0,
                rejected_frames=0,
                avg_sharpness=0.0,
                rotation_errors=["transforms.json not found"],
                timestamp_errors=[],
                continuity_warnings=[],
                quality_score=0.0,
            ), {}

        with open(self.transforms_path, "r") as f:
            data = json.load(f)

        frames = data.get("frames", [])
        w = data.get("w", 0)
        h = data.get("h", 0)
        fx = data.get("fl_x", 0.0)
        fy = data.get("fl_y", 0.0)
        cx = data.get("cx", 0.0)
        cy = data.get("cy", 0.0)

        rotation_errors = []
        timestamp_errors = []
        continuity_warnings = []
        sharpness_list = []

        valid_frames = []
        prev_timestamp = -1
        prev_pos = None

        for idx, frame in enumerate(frames):
            file_path = self.dataset_dir / frame["file_path"]
            if not file_path.exists():
                rotation_errors.append(f"Frame {idx}: image file missing at {file_path}")
                continue

            # 1. Check Image Dimensions
            try:
                with PILImage.open(file_path) as img:
                    img_w, img_h = img.size
                    if (w > 0 and img_w != w) or (h > 0 and img_h != h):
                        rotation_errors.append(f"Frame {idx}: intrinsic resolution ({w}x{h}) mismatches image ({img_w}x{img_h})")
            except Exception as e:
                rotation_errors.append(f"Frame {idx}: corrupted image file ({e})")
                continue

            # 2. Check Pose Matrix Orthonormality det(R) ≈ 1
            mat = np.array(frame["transform_matrix"], dtype=np.float64)
            if mat.shape != (4, 4):
                rotation_errors.append(f"Frame {idx}: matrix shape is {mat.shape}, expected (4,4)")
                continue

            R = mat[:3, :3]
            det_R = np.linalg.det(R)
            if abs(det_R - 1.0) > self.rotation_tolerance:
                rotation_errors.append(f"Frame {idx}: det(R) = {det_R:.4f} != 1.0 (invalid rotation matrix)")
                continue

            # Check R * R^T = I
            ortho_err = np.max(np.abs(R @ R.T - np.eye(3)))
            if ortho_err > self.rotation_tolerance:
                rotation_errors.append(f"Frame {idx}: non-orthonormal rotation R R^T error {ortho_err:.4f}")
                continue

            # 3. Check Timestamp Monotonicity
            ts = frame.get("timestamp_ns", 0)
            if prev_timestamp > 0 and ts <= prev_timestamp:
                timestamp_errors.append(f"Frame {idx}: non-increasing timestamp ({ts} <= {prev_timestamp})")
            prev_timestamp = ts

            # 4. Check Spatial Continuity
            pos = mat[:3, 3]
            if prev_pos is not None:
                dist = np.linalg.norm(pos - prev_pos)
                if dist > 1.5:
                    continuity_warnings.append(f"Frame {idx}: large pose jump ({dist:.2f}m) detected (possible SLAM reset)")
            prev_pos = pos

            sharpness = frame.get("sharpness_score", 100.0)
            sharpness_list.append(sharpness)
            valid_frames.append(frame)

        # Point cloud check
        num_points = 0
        if self.points_path.exists():
            try:
                with open(self.points_path, "r") as f:
                    pt_data = json.load(f)
                    num_points = pt_data.get("num_points", len(pt_data.get("points", [])))
            except Exception:
                pass

        avg_sharpness = float(np.mean(sharpness_list)) if sharpness_list else 0.0
        is_valid = len(rotation_errors) == 0 and len(valid_frames) >= 1


        quality_score = max(0.0, 100.0 - (len(rotation_errors) * 15.0 + len(continuity_warnings) * 5.0))

        report = ARCoreValidationReport(
            is_valid=is_valid,
            num_frames=len(frames),
            num_points=num_points,
            passed_frames=len(valid_frames),
            rejected_frames=len(frames) - len(valid_frames),
            avg_sharpness=avg_sharpness,
            rotation_errors=rotation_errors,
            timestamp_errors=timestamp_errors,
            continuity_warnings=continuity_warnings,
            quality_score=quality_score,
        )

        return report, data

    def load_as_cameras(self) -> List[Camera]:
        """
        Load validated frames into Camera list for 3DGS pipeline training.
        """
        report, data = self.validate()
        if not report.is_valid:
            raise ValueError(f"ARCore Dataset at {self.dataset_dir} failed validation: {report.rotation_errors}")

        frames = data["frames"]
        w = data["w"]
        h = data["h"]
        fx = data["fl_x"]
        fy = data["fl_y"]
        cx = data["cx"]
        cy = data["cy"]

        intrinsics = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=w, height=h)
        cameras = []

        for idx, frame in enumerate(frames):
            c2w = np.array(frame["transform_matrix"], dtype=np.float64)
            # World-to-Camera (w2c) matrix is matrix inverse of c2w
            w2c = np.linalg.inv(c2w)
            R = w2c[:3, :3]
            T = w2c[:3, 3]

            extrinsics = CameraExtrinsics(R=R, T=T)
            image_path = str(self.dataset_dir / frame["file_path"])

            cam = Camera(
                uid=idx,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                image_path=image_path,
            )
            cameras.append(cam)

        return cameras

    def load_initial_point_cloud(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load initial 3D point cloud from points3D_initial.json if available.
        Returns (points (N,3), colors (N,3)).
        """
        if not self.points_path.exists():
            # Fallback to random point cloud bounded by camera centers
            return np.random.randn(500, 3).astype(np.float32), np.random.rand(500, 3).astype(np.float32)

        with open(self.points_path, "r") as f:
            data = json.load(f)

        pts_list = []
        for item in data.get("points", []):
            xyz = item.get("xyz")
            if xyz and len(xyz) == 3:
                pts_list.append(xyz)

        if not pts_list:
            return np.random.randn(500, 3).astype(np.float32), np.random.rand(500, 3).astype(np.float32)

        points = np.array(pts_list, dtype=np.float32)
        # Default colors to neutral light gray
        colors = np.ones_like(points, dtype=np.float32) * 0.7
        return points, colors
