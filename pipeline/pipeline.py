"""
End-to-end 3D Gaussian Splatting pipeline.

Orchestrates: COLMAP loading → Gaussian initialization → training → evaluation → saving.
"""
from __future__ import annotations
import os
import math
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch


class Pipeline:
    """
    Full 3DGS pipeline from COLMAP sparse reconstruction to trained model.

    Usage:
        pipeline = Pipeline(config={
            'source_path': '/path/to/colmap',
            'images_path': '/path/to/images',
            'model_path':  '/path/to/output',
            'sh_degree':   3,
        })
        scene = pipeline.load_scene()
        gaussians = pipeline.initialize_gaussians(scene)
        gaussians = pipeline.train(scene, output_dir='./output')
        pipeline.render_views(gaussians, scene.test_cameras, './output/renders')
        metrics = pipeline.evaluate(gaussians, scene.test_cameras)
    """

    def __init__(self, config: dict):
        self.config = config
        self.sh_degree = config.get("sh_degree", 3)
        self.source_path = Path(config.get("source_path", "."))
        self.images_path = Path(config.get("images_path", self.source_path / "images"))
        self.model_path = Path(config.get("model_path", "./output"))
        self.white_bg = config.get("white_background", False)

    # -----------------------------------------------------------------------
    # Scene loading
    # -----------------------------------------------------------------------

    def load_scene(
        self,
        colmap_path: Optional[str] = None,
        images_path: Optional[str] = None,
        test_fraction: float = 0.1,
    ):
        """
        Load a COLMAP sparse reconstruction and return a Scene.

        Args:
            colmap_path:   path to COLMAP sparse dir (default: source_path/sparse/0)
            images_path:   path to images dir
            test_fraction: fraction of cameras to use for evaluation

        Returns:
            Scene
        """
        from .colmap_loader import ColmapSceneLoader
        from .scene import Scene

        cp = Path(colmap_path) if colmap_path else self.source_path / "sparse" / "0"
        ip = Path(images_path) if images_path else self.images_path

        print(f"Loading COLMAP from: {cp}")
        print(f"Images from:        {ip}")

        loader = ColmapSceneLoader()
        cameras, points3d, colors = loader.load(str(cp), str(ip))

        print(f"Loaded {len(cameras)} cameras, {len(points3d)} 3D points")

        # Train/test split: take every Nth camera as test
        n_test = max(1, int(len(cameras) * test_fraction))
        test_idx = set(range(0, len(cameras), max(1, len(cameras) // n_test)))
        train_cameras = [c for i, c in enumerate(cameras) if i not in test_idx]
        test_cameras  = [c for i, c in enumerate(cameras) if i in test_idx]

        scene = Scene(
            train_cameras=train_cameras,
            test_cameras=test_cameras,
            points3d=points3d,
            colors=colors,
        )
        print(f"Scene: {scene}")
        return scene

    # -----------------------------------------------------------------------
    # Gaussian initialization
    # -----------------------------------------------------------------------

    def initialize_gaussians(self, scene, device: str = "cuda"):
        """
        Initialize GaussianModel from scene point cloud.

        Args:
            scene:  Scene instance
            device: torch device

        Returns:
            GaussianModel on given device
        """
        from ..core.gaussians import GaussianModel

        gaussians = GaussianModel(sh_degree=self.sh_degree)
        gaussians.init_from_pointcloud(
            points=scene.points3d,
            colors=scene.colors,
        )
        gaussians = gaussians.to(device) if hasattr(gaussians, 'to') else gaussians
        # Move parameters to device
        for name, param in gaussians.named_parameters():
            param.data = param.data.to(device)
        gaussians.xyz_gradient_accum = gaussians.xyz_gradient_accum.to(device)
        gaussians.denom = gaussians.denom.to(device)
        gaussians.max_radii2D = gaussians.max_radii2D.to(device)
        print(f"Initialized GaussianModel: {gaussians}")
        return gaussians

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def train(
        self,
        scene,
        output_dir: Optional[str] = None,
        iterations: int = 30_000,
        callbacks: List[Callable] = [],
        device: str = "cuda",
    ):
        """
        Full training run.

        Args:
            scene:      Scene with train/test cameras and point cloud
            output_dir: where to save checkpoints and PLY files
            iterations: number of training iterations
            callbacks:  list of (iteration, metrics) callbacks
            device:     'cuda' or 'cpu'

        Returns:
            trained GaussianModel
        """
        from ..training import GaussianTrainer, TrainingConfig
        from ..renderer import TileBasedRasterizer

        out = Path(output_dir) if output_dir else self.model_path
        out.mkdir(parents=True, exist_ok=True)

        cfg = TrainingConfig(
            iterations=iterations,
            model_path=str(out),
            white_background=self.white_bg,
            sh_degree=self.sh_degree,
        )
        cfg.save_json(out / "config.json")

        gaussians = self.initialize_gaussians(scene, device=device)
        renderer  = TileBasedRasterizer()
        trainer   = GaussianTrainer(cfg)

        train_cams = [c.to(device) for c in scene.train_cameras]
        test_cams  = [c.to(device) for c in scene.test_cameras]
        trainer.setup(gaussians, train_cams, test_cams)

        t0 = time.time()
        print(f"Starting training for {iterations} iterations...")
        trained = trainer.train(renderer, callbacks=callbacks)

        elapsed = time.time() - t0
        print(f"Training complete in {elapsed/60:.1f} min")

        # Save final model
        final_ply = out / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
        trained.save_ply(final_ply)

        return trained

    # -----------------------------------------------------------------------
    # Rendering
    # -----------------------------------------------------------------------

    def render_views(
        self,
        gaussians,
        cameras: List,
        output_dir: str,
        device: str = "cuda",
    ) -> List[str]:
        """
        Render all cameras and save images.

        Returns:
            list of saved image paths
        """
        from ..renderer import TileBasedRasterizer
        from PIL import Image as PILImage
        import torchvision.transforms.functional as TF

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        renderer = TileBasedRasterizer()
        bg = torch.ones(3, device=device) if self.white_bg else torch.zeros(3, device=device)

        saved = []
        for i, camera in enumerate(cameras):
            camera = camera.to(device)
            with torch.no_grad():
                result = renderer.render(gaussians, camera, bg_color=bg)
            img = result["render"].clamp(0, 1)  # (3, H, W)
            img_pil = TF.to_pil_image(img.cpu())
            path = out / f"{i:04d}.png"
            img_pil.save(str(path))
            saved.append(str(path))
            print(f"  Saved render {i+1}/{len(cameras)}: {path}")

        return saved

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------

    def evaluate(
        self,
        gaussians,
        cameras: List,
        device: str = "cuda",
    ) -> Dict[str, float]:
        """
        Compute PSNR and SSIM on given cameras.

        Returns:
            dict with 'psnr' (dB), 'ssim' (higher = better), 'n_views'
        """
        from ..renderer import TileBasedRasterizer
        from ..training.loss import ssim_loss
        from ..training.trainer import GaussianTrainer

        renderer = TileBasedRasterizer()
        bg = torch.ones(3, device=device) if self.white_bg else torch.zeros(3, device=device)

        psnrs, ssims = [], []
        for camera in cameras:
            camera = camera.to(device)
            gt = camera.load_image().to(device)

            with torch.no_grad():
                result = renderer.render(gaussians, camera, bg_color=bg)
            pred = result["render"].clamp(0, 1)

            if gt.shape != pred.shape:
                import torch.nn.functional as F
                gt = F.interpolate(gt.unsqueeze(0), size=pred.shape[-2:], mode='bilinear').squeeze(0)

            psnrs.append(GaussianTrainer.compute_psnr(pred, gt))
            ssims.append(1.0 - ssim_loss(pred, gt).item())

        return {
            "psnr":    float(np.mean(psnrs)),
            "ssim":    float(np.mean(ssims)),
            "n_views": len(cameras),
        }
