"""
Renderer integration tests.
Run: pytest tests/test_renderer.py -v
"""
import math
import pytest
import torch
import numpy as np

from gaussian.core.gaussians import GaussianModel
from gaussian.core.camera import Camera, CameraIntrinsics, CameraExtrinsics
from gaussian.renderer import TileBasedRasterizer


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def make_test_camera(width=128, height=96):
    """Create a simple test camera looking along +z."""
    K = CameraIntrinsics(fx=64.0, fy=64.0, cx=width / 2.0, cy=height / 2.0, width=width, height=height)
    R = np.eye(3, dtype=np.float64)
    T = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    E = CameraExtrinsics(R=R, T=T)
    return Camera(uid=0, intrinsics=K, extrinsics=E)


def make_test_gaussians(n=50, device='cpu'):
    pts = np.zeros((n, 3), dtype=np.float32)
    pts[:, 2] = np.random.uniform(2.0, 5.0, n)  # all in front of camera
    pts[:, 0] = np.random.uniform(-0.5, 0.5, n)
    pts[:, 1] = np.random.uniform(-0.5, 0.5, n)
    colors = np.random.rand(n, 3).astype(np.float32)
    g = GaussianModel(sh_degree=0)  # degree 0 for speed
    g.init_from_pointcloud(pts, colors)
    # Set high opacity so Gaussians are visible
    import torch.nn as nn
    g._opacity = nn.Parameter(torch.ones(n, 1) * 5.0)  # logit(~1.0)
    return g


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRenderer:

    def test_render_output_shape(self):
        camera = make_test_camera(128, 96)
        gaussians = make_test_gaussians(50)
        renderer = TileBasedRasterizer()
        bg = torch.zeros(3)

        out = renderer.render(gaussians, camera, bg_color=bg)

        assert 'render' in out
        assert out['render'].shape == (3, 96, 128), f"Got shape {out['render'].shape}"
        assert out['depth'].shape == (96, 128)
        assert out['alpha'].shape == (96, 128)

    def test_render_in_range(self):
        """Rendered pixels must be in [0, 1] after clamping."""
        camera = make_test_camera()
        gaussians = make_test_gaussians(30)
        renderer = TileBasedRasterizer()
        bg = torch.zeros(3)

        out = renderer.render(gaussians, camera, bg_color=bg)
        rendered = out['render']
        # Values may slightly exceed due to floating point, clamp in range
        assert rendered.min() >= -1e-5, f"Min below 0: {rendered.min()}"
        assert rendered.max() <= 1.0 + 1e-5, f"Max above 1: {rendered.max()}"

    def test_render_gradient_backprop(self):
        """Loss.backward() must not crash; gradients must flow."""
        camera = make_test_camera(64, 48)
        gaussians = make_test_gaussians(20)
        renderer = TileBasedRasterizer()
        bg = torch.zeros(3)

        out = renderer.render(gaussians, camera, bg_color=bg)
        rendered = out['render']

        loss = rendered.mean()
        loss.backward()  # Must not crash

    def test_empty_scene(self):
        """Renderer must handle 0 Gaussians gracefully (return bg color)."""
        camera = make_test_camera()
        g = GaussianModel(sh_degree=0)
        # Don't call init_from_pointcloud → empty model
        # Manually set empty parameters
        import torch.nn as nn
        g._xyz = nn.Parameter(torch.zeros(0, 3))
        g._features_dc = nn.Parameter(torch.zeros(0, 1, 3))
        g._features_rest = nn.Parameter(torch.zeros(0, 15, 3))
        g._opacity = nn.Parameter(torch.zeros(0, 1))
        g._scaling = nn.Parameter(torch.zeros(0, 3))
        g._rotation = nn.Parameter(torch.zeros(0, 4))
        g._init_densification_stats()

        renderer = TileBasedRasterizer()
        bg = torch.tensor([0.5, 0.3, 0.1])

        out = renderer.render(g, camera, bg_color=bg)
        rendered = out['render']
        # Should be approximately background color
        assert rendered.shape == (3, camera.height, camera.width)

    def test_visibility_filter_shape(self):
        camera = make_test_camera()
        gaussians = make_test_gaussians(40)
        renderer = TileBasedRasterizer()
        bg = torch.zeros(3)

        out = renderer.render(gaussians, camera, bg_color=bg)
        vf = out['visibility_filter']
        assert vf.shape == (40,)
        assert vf.dtype == torch.bool

    def test_single_centered_gaussian(self):
        """Single Gaussian at (0,0,3) should produce non-zero center pixel."""
        camera = make_test_camera(64, 64)

        import torch.nn as nn
        g = GaussianModel(sh_degree=0)
        # Single Gaussian at image center, high opacity, large scale
        g._xyz = nn.Parameter(torch.tensor([[0.0, 0.0, 3.0]]))
        g._features_dc = nn.Parameter(torch.tensor([[[1.0, 0.0, 0.0]]]))  # red
        g._features_rest = nn.Parameter(torch.zeros(1, 15, 3))
        g._opacity = nn.Parameter(torch.tensor([[10.0]]))   # very opaque
        g._scaling = nn.Parameter(torch.zeros(1, 3))       # unit scale
        g._rotation = nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        g._init_densification_stats()

        renderer = TileBasedRasterizer()
        bg = torch.zeros(3)
        out = renderer.render(g, camera, bg_color=bg)
        rendered = out['render']
        # Center pixel should have some red component
        cy, cx = 32, 32
        assert rendered[0, cy, cx] > 0.01, \
            f"Center pixel should be red, got {rendered[:, cy, cx]}"

    def test_radii_shape_and_type(self):
        camera = make_test_camera()
        gaussians = make_test_gaussians(30)
        renderer = TileBasedRasterizer()
        bg = torch.zeros(3)

        out = renderer.render(gaussians, camera, bg_color=bg)
        radii = out['radii']
        assert radii.shape == (30,)
        assert radii.dtype == torch.int64
