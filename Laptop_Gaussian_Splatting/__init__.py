"""
gaussian — Production-grade 3D Gaussian Splatting system.

Modules:
  core/        Mathematical primitives, camera model, GaussianModel
  renderer/    Differentiable tile-based rasterizer
  training/    Loss functions, trainer, config
  pipeline/    COLMAP loader, scene, end-to-end pipeline
  experimental/ LOD, temporal dynamics, neural balancer, self-optimizer
  scene/       Spatial hash, point cloud I/O
  ui/          Interactive viewer, video export
  tests/       Unit tests and benchmarks
"""

__version__ = "1.0.0"
__author__ = "3DGS Research System"

from .core import GaussianModel, Camera
from .pipeline import Pipeline
from .renderer import TileBasedRasterizer
from .training import GaussianTrainer, TrainingConfig

__all__ = [
    "GaussianModel",
    "Camera",
    "Pipeline",
    "TileBasedRasterizer",
    "GaussianTrainer",
    "TrainingConfig",
]
