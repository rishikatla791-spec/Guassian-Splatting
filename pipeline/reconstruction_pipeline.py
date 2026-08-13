"""
High-Fidelity Image-to-3D Reconstruction Pipeline.

Orchestrates the 9-stage 3D reconstruction pipeline:
  Images ──► Camera/Pose Estimation ──► Multi-View Matching ──► Dense Geometry
  ──► High-Resolution Appearance ──► 3D Reconstruction ──► Refinement
  ──► Gaussian Splatting ──► Photorealistic Novel Views
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .pose_estimation import PoseEstimator
from .background_masker import ObjectMaskGenerator
from .dense_geometry import DenseGeometryReconstructor
from .validation_suite import ValidationSuite
from .memory_optimizer import VRAMBudgetManager
from ..core.gaussians import GaussianModel
from ..renderer import TileBasedRasterizer
from ..training import GaussianTrainer, TrainingConfig


class ReconstructionPipeline:
    """
    End-to-End High-Fidelity Multi-View Image-to-3D Reconstruction Pipeline.
    """

    def __init__(self, config: dict):
        self.config = config
        self.images_path = Path(config.get("images_path", "./imgaes"))
        self.output_dir = Path(config.get("output_dir", "./output_reconstruction"))
        self.sh_degree = config.get("sh_degree", 3)
        self.iterations = config.get("iterations", 15_000)
        self.backend_pref = config.get("backend", "auto")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize sub-modules
        self.pose_estimator = PoseEstimator(backend_preference=self.backend_pref)
        self.mask_generator = ObjectMaskGenerator()
        self.geometry_reconstructor = DenseGeometryReconstructor()
        self.validation_suite = ValidationSuite(device=self.device)
        self.memory_manager = VRAMBudgetManager(max_vram_gb=5.2)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_full_pipeline(self) -> Tuple[GaussianModel, Dict[str, float]]:
        """
        Execute the complete 9-stage Image-to-3D Reconstruction Pipeline.

        Returns:
            trained_gaussians: GaussianModel
            validation_metrics: Dict[str, float]
        """
        t_start = time.time()
        print("\n==========================================================================")
        print("  HIGH-FIDELITY MULTI-VIEW IMAGE-TO-3D RECONSTRUCTION PIPELINE STARTED")
        print("==========================================================================\n")

        # STAGE 1 & 2 & 3: Pose Estimation & Multi-View Matching
        print("[Stage 1-3] Camera Pose Estimation & Multi-View Feature Matching...")
        cameras, raw_points3d, raw_colors3d, pose_metrics = self.pose_estimator.estimate_poses(
            images_dir=self.images_path
        )
        self.memory_manager.print_memory_status("Post Pose Estimation")

        # STAGE 4: Background Masking & Foreground Object Isolation
        print("\n[Stage 4] Background Object Segmentation & Silhouette Isolation...")
        cameras = self.mask_generator.process_camera_list(cameras)

        # STAGE 5: Dense Geometry & Surface Normal Triangulation
        print("\n[Stage 5] Dense Geometry Reconstruction & Surface Normal Triangulation...")
        clean_pts, clean_cols, normals, knn_dist = self.geometry_reconstructor.process_point_cloud(
            points=raw_points3d,
            colors=raw_colors3d
        )

        # Train/Test Split (85% Train / 15% Validation Holdout)
        n_test = max(1, int(len(cameras) * 0.15))
        test_indices = set(range(0, len(cameras), max(1, len(cameras) // n_test)))
        train_cameras = [c for i, c in enumerate(cameras) if i not in test_indices]
        test_cameras  = [c for i, c in enumerate(cameras) if i in test_indices]

        print(f"[OK] Split Dataset: {len(train_cameras)} Training Cameras | {len(test_cameras)} Validation Holdout Cameras")

        # STAGE 6: Gaussian Initialization
        print("\n[Stage 6] Initializing 3D Gaussian Scene Representation...")
        gaussians = GaussianModel(sh_degree=self.sh_degree)
        gaussians.init_from_pointcloud(clean_pts, clean_cols)
        gaussians = gaussians.to(self.device)

        # STAGE 7 & 8: 3DGS Training, Densification, Pruning & Appearance Refinement
        print("\n[Stage 7-8] 3D Gaussian Splatting Training & Appearance Refinement...")
        cfg = TrainingConfig(
            iterations=self.iterations,
            model_path=str(self.output_dir),
            sh_degree=self.sh_degree,
            white_background=False,
            random_background=False,
        )

        renderer = TileBasedRasterizer()
        trainer = GaussianTrainer(cfg)

        train_cams = [c.to(self.device) for c in train_cameras]
        test_cams  = [c.to(self.device) for c in test_cameras]
        trainer.setup(gaussians, train_cams, test_cams)

        trained_gaussians = trainer.train(renderer)

        self.memory_manager.optimize_memory(force=True)

        # STAGE 9: Validation & Photorealistic Novel View Evaluation
        print("\n[Stage 9] Comprehensive Validation & Quantitative Metrics Evaluation...")
        val_metrics = self.validation_suite.evaluate(
            gaussians=trained_gaussians,
            renderer=renderer,
            test_cameras=test_cams,
            points3d_gt=clean_pts,
        )

        # Save Final Model PLY
        final_ply = self.output_dir / "point_cloud" / f"iteration_{self.iterations}" / "point_cloud.ply"
        final_ply.parent.mkdir(parents=True, exist_ok=True)
        trained_gaussians.save_ply(str(final_ply))
        print(f"\n[OK] Saved final reconstructed point cloud model: {final_ply}")

        # STAGE 10: Real 3D Mesh Generation & Multi-Format Export (OBJ, GLTF, PLY Mesh)
        print("\n[Stage 10] Real 3D Surface Mesh Generation & Multi-Format Export (OBJ, GLTF, PLY)...")
        try:
            from ..export_3d_mesh import export_all_formats
            mesh_paths = export_all_formats(final_ply, self.output_dir)
            print(f"[OK] Generated Wavefront 3D OBJ model: {mesh_paths['obj']}")
            print(f"[OK] Generated 3D Polygonal PLY mesh:   {mesh_paths['ply_mesh']}")
            print(f"[OK] Generated Standard GLTF 2.0 asset:  {mesh_paths['gltf']}")
        except Exception as e:
            print(f"[Mesh Export Warning] Could not export polygonal mesh: {e}")

        t_elapsed = time.time() - t_start
        print(f"\n==========================================================================")
        print(f"  RECONSTRUCTION PIPELINE COMPLETED IN {t_elapsed/60.0:.2f} MINUTES")
        print(f"==========================================================================\n")

        return trained_gaussians, val_metrics

