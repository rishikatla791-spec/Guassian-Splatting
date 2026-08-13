"""
Comprehensive Unit Test Suite for Next-Gen Experimental Features.

Tests:
- SelfOptimizingAllocator
- TemporalGaussianEvolution
- HierarchicalLOD
- NeuralQualityBalancer
- PredictiveViewStreamer
- SemanticGaussianExtension
- AdaptiveFPSController
- GaussianSceneCompressor
"""

import math
import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch

from gaussian.core.gaussians import GaussianModel
from gaussian.core.camera import Camera, CameraIntrinsics, CameraExtrinsics

from gaussian.experimental.self_optimizer import SelfOptimizingAllocator
from gaussian.experimental.temporal import TemporalGaussianEvolution
from gaussian.experimental.lod import HierarchicalLOD
from gaussian.experimental.neural_balance import NeuralQualityBalancer
from gaussian.experimental.predictive_streaming import PredictiveViewStreamer
from gaussian.experimental.semantic import SemanticGaussianExtension
from gaussian.experimental.adaptive_fps import AdaptiveFPSController
from gaussian.experimental.compression import GaussianSceneCompressor
from gaussian.renderer.tile_rasterizer import TileBasedRasterizer


def make_test_scene(n=100):
    pts = np.random.randn(n, 3).astype(np.float32)
    pts[:, 2] += 4.0
    colors = np.random.rand(n, 3).astype(np.float32)
    g = GaussianModel(sh_degree=1)
    g.init_from_pointcloud(pts, colors)
    return g


def make_test_camera():
    K = CameraIntrinsics(fx=64.0, fy=64.0, cx=32.0, cy=32.0, width=64, height=64)
    E = CameraExtrinsics(R=np.eye(3), T=np.zeros(3))
    return Camera(uid=0, intrinsics=K, extrinsics=E)


class TestExperimentalFeatures:

    def test_self_optimizing_allocator(self):
        g = make_test_scene(100)
        allocator = SelfOptimizingAllocator(budget=50)

        scores = allocator.update_importance(g)
        assert scores.shape == (100,)
        assert (scores >= 0).all()

        import torch.optim as optim
        optimizer = optim.Adam(g.parameters(), lr=1e-3)
        allocator.reallocate(g, optimizer, target_count=50)
        assert g.num_gaussians == 50

    def test_temporal_gaussian_evolution(self):
        g = make_test_scene(50)
        temporal = TemporalGaussianEvolution(g)

        pos_before = g.get_xyz.clone()
        temporal.v_xyz[:, 2] = 1.0  # Set forward velocity
        temporal.evolve(dt=0.1)

        pos_after = g.get_xyz
        assert not torch.allclose(pos_before, pos_after)
        assert torch.allclose(pos_after[:, 2], pos_before[:, 2] + 0.1)

        loss = temporal.compute_velocity_regularization()
        assert loss.item() > 0.0

    def test_hierarchical_lod(self):
        g = make_test_scene(100)
        lod = HierarchicalLOD()
        lod.build(g, levels=3)

        assert lod._n_levels == 3
        level_near = lod.get_lod_level(camera_distance=0.5)
        level_far = lod.get_lod_level(camera_distance=100.0)
        assert level_near <= level_far

    def test_neural_quality_balancer(self):
        g = make_test_scene(40)
        balancer = NeuralQualityBalancer()

        feats = balancer.extract_features(g)
        assert feats.shape == (40, 4)

        scores = balancer.predict(g)
        assert scores.shape == (40,)
        assert (scores >= 0).all() and (scores <= 1.0).all()

    def test_predictive_view_streamer(self):
        cam = make_test_camera()
        streamer = PredictiveViewStreamer()

        # Push camera history
        for i in range(4):
            cam_copy = make_test_camera()
            cam_copy.extrinsics.T[0] = i * 0.1
            streamer.push_pose(cam_copy, timestamp=i * 0.033)

        pred_cam = streamer.predict_future_camera(cam, dt_lookahead=0.033)
        assert pred_cam.extrinsics.T[0] > 0.3  # Extrapolated forward

    def test_semantic_gaussian_extension(self):
        g = make_test_scene(30)
        semantic_ext = SemanticGaussianExtension(num_classes=5)

        semantic_ext.attach_semantics(g)
        assert hasattr(g, "semantics")
        assert g.semantics.shape == (30, 5)

        probs = semantic_ext.get_semantic_probabilities(g)
        assert probs.shape == (30, 5)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(30))

        mask = semantic_ext.filter_by_class(g, target_class=0, threshold=0.1)
        assert mask.shape == (30,)

    def test_adaptive_fps_controller(self):
        controller = AdaptiveFPSController(target_fps=60.0)

        # Simulate slow frame (30ms = 33 FPS)
        status_slow = controller.update(frame_latency_s=0.030)
        assert status_slow["sh_degree"] <= 3

        # Simulate fast frames
        for _ in range(5):
            status_fast = controller.update(frame_latency_s=0.005)

    def test_gaussian_scene_compression(self):
        g = make_test_scene(80)
        compressor = GaussianSceneCompressor(codebook_size=32)

        payload = compressor.compress(g)
        assert payload["num_gaussians"] == 80
        assert payload["opacities_uint8"].dtype == np.uint8

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "scene.gsp"
            compressor.save_gsp(payload, file_path)
            assert file_path.exists()

            loaded_payload = compressor.load_gsp(file_path)
            g_decompressed = compressor.decompress(loaded_payload)

            assert g_decompressed.num_gaussians == 80
            assert g_decompressed._xyz.shape == g._xyz.shape
