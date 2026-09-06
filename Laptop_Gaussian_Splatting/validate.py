import sys
from pathlib import Path

# Add current directory and parent directory to python path for robust importing
root_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir.parent))

# Test core imports
try:
    from core.math_utils import quaternion_to_rotation_matrix, build_covariance_3d, build_covariance_2d
    from core.sh import eval_sh, RGB2SH, SH2RGB
    from core.camera import Camera, CameraIntrinsics, CameraExtrinsics
    from core.gaussians import GaussianModel
    from core.cuda_ops import build_covariance_3d_cuda, build_covariance_2d_cuda, invert_cov2d_cuda, compute_radius_cuda
    print('core & CUDA ops: OK')
except ImportError:
    from gaussian.core.math_utils import quaternion_to_rotation_matrix, build_covariance_3d, build_covariance_2d
    from gaussian.core.sh import eval_sh, RGB2SH, SH2RGB
    from gaussian.core.camera import Camera, CameraIntrinsics, CameraExtrinsics
    from gaussian.core.gaussians import GaussianModel
    from gaussian.core.cuda_ops import build_covariance_3d_cuda, build_covariance_2d_cuda, invert_cov2d_cuda, compute_radius_cuda
    print('core & CUDA ops (packaged): OK')

# Test renderer imports
try:
    from renderer.gaussian_rasterizer import GaussianRasterizer, RasterizationSettings
    from renderer.cuda_rasterizer import CUDAGaussianRasterizer
    from renderer.tile_rasterizer import TileBasedRasterizer
    from renderer.visibility import VisibilityCuller
    print('renderer & CUDA rasterizer: OK')
except ImportError:
    from gaussian.renderer.gaussian_rasterizer import GaussianRasterizer, RasterizationSettings
    from gaussian.renderer.cuda_rasterizer import CUDAGaussianRasterizer
    from gaussian.renderer.tile_rasterizer import TileBasedRasterizer
    from gaussian.renderer.visibility import VisibilityCuller
    print('renderer & CUDA rasterizer (packaged): OK')

# Test training imports
try:
    from training.config import TrainingConfig
    from training.loss import l1_loss, ssim_loss, combined_loss
    from training.trainer import GaussianTrainer
    print('training: OK')
except ImportError:
    from gaussian.training.config import TrainingConfig
    from gaussian.training.loss import l1_loss, ssim_loss, combined_loss
    from gaussian.training.trainer import GaussianTrainer
    print('training (packaged): OK')

# Test pipeline imports
try:
    from pipeline.scene import Scene
    from pipeline.pipeline import Pipeline
    from pipeline.colmap_loader import ColmapSceneLoader
    from pipeline.pose_estimation import PoseEstimator
    from pipeline.background_masker import ObjectMaskGenerator
    from pipeline.dense_geometry import DenseGeometryReconstructor
    from pipeline.validation_suite import ValidationSuite
    from pipeline.memory_optimizer import VRAMBudgetManager
    from pipeline.reconstruction_pipeline import ReconstructionPipeline
    print('pipeline & reconstruction components: OK')
except ImportError:
    from gaussian.pipeline.scene import Scene
    from gaussian.pipeline.pipeline import Pipeline
    from gaussian.pipeline.colmap_loader import ColmapSceneLoader
    from gaussian.pipeline.pose_estimation import PoseEstimator
    from gaussian.pipeline.background_masker import ObjectMaskGenerator
    from gaussian.pipeline.dense_geometry import DenseGeometryReconstructor
    from gaussian.pipeline.validation_suite import ValidationSuite
    from gaussian.pipeline.memory_optimizer import VRAMBudgetManager
    from gaussian.pipeline.reconstruction_pipeline import ReconstructionPipeline
    print('pipeline & reconstruction components (packaged): OK')

# Test experimental imports
try:
    from experimental.lod import HierarchicalLOD
    from experimental.temporal import TemporalGaussianEvolution
    from experimental.neural_balance import NeuralQualityBalancer
    from experimental.self_optimizer import SelfOptimizingAllocator
    print('experimental: OK')
except ImportError:
    from gaussian.experimental.lod import HierarchicalLOD
    from gaussian.experimental.temporal import TemporalGaussianEvolution
    from gaussian.experimental.neural_balance import NeuralQualityBalancer
    from gaussian.experimental.self_optimizer import SelfOptimizingAllocator
    print('experimental (packaged): OK')

print()
print('ALL IMPORTS SUCCESSFUL!')

# Quick functional test: GaussianModel + math + validation
import torch
import numpy as np

pts = np.random.randn(500, 3).astype('float32')
pts[:, 2] += 5.0
colors = np.random.rand(500, 3).astype('float32')
g = GaussianModel(sh_degree=3)
g.init_from_pointcloud(pts, colors)
print(f'GaussianModel init: {g}')

# Validation suite chamfer & PSNR test
val_suite = ValidationSuite(device='cpu')
img1 = torch.rand(3, 128, 128)
img2 = torch.rand(3, 128, 128)
psnr_val = val_suite.compute_psnr(img1, img2)
ssim_val = val_suite.compute_ssim(img1, img2)
chamfer_val = val_suite.compute_chamfer_distance(pts, pts + 0.05)

print(f'Validation metrics test: PSNR={psnr_val:.2f} dB, SSIM={ssim_val:.4f}, Chamfer={chamfer_val:.6f}')

# VRAM Manager test
vram_mgr = VRAMBudgetManager()
stats = vram_mgr.get_vram_stats()
print(f'VRAM Manager test: allocated={stats["allocated_gb"]:.2f} GB')

print()
print('=== ALL PIPELINE VALIDATION TESTS PASSED ===')
