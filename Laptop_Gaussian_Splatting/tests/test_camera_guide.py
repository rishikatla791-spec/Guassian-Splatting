"""
Unit tests for Camera Guidance & View Coverage Planner.
"""

import numpy as np
import pytest

from gaussian.core.camera import Camera, CameraIntrinsics, CameraExtrinsics
from gaussian.pipeline.camera_guide import ViewCoveragePlanner


def test_view_planner_initialization():
    planner = ViewCoveragePlanner(center=np.array([0.0, 0.0, 0.0]), radius=2.0, num_elevation_rings=3, samples_per_ring=12)
    assert len(planner.target_nodes) == 36
    assert planner.target_nodes[0].position.shape == (3,)


def test_view_planner_evaluation():
    planner = ViewCoveragePlanner(center=np.array([0.0, 0.0, 0.0]), radius=2.0, num_elevation_rings=2, samples_per_ring=8)
    
    # Create mock camera looking at target
    eye = np.array([2.0, 0.5, 0.0])
    target = np.array([0.0, 0.0, 0.0])
    ext = CameraExtrinsics.from_look_at(eye, target)
    intr = CameraIntrinsics(fx=800, fy=800, cx=400, cy=300, width=800, height=600)
    cam = Camera(uid=1, intrinsics=intr, extrinsics=ext)


    metrics = planner.evaluate_captured_cameras([cam])
    assert "coverage_ratio" in metrics
    assert "quality_score" in metrics
    assert metrics["satisfied_nodes"] >= 1


def test_next_best_view_recommendations():
    planner = ViewCoveragePlanner(center=np.array([0.0, 0.0, 0.0]), radius=2.0, num_elevation_rings=2, samples_per_ring=6)
    nbv = planner.get_next_best_views(top_k=3)
    assert len(nbv) <= 3
