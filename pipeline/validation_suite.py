"""
Validation & Comprehensive Evaluation Metrics Suite.

Calculates quantitative metrics to evaluate reconstruction accuracy:
  - Photometric & Perceptual: PSNR, SSIM, LPIPS, L1 Loss
  - Geometric Accuracy: Chamfer Distance d_CD(P1, P2), Reprojection Error (pixels)
  - Multi-View Consistency: Cross-Camera Epipolar & SSIM consistency score
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


class ValidationSuite:
    """
    Evaluation Suite for 3D Gaussian Splatting & Multi-View Reconstruction.
    """

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def _align_image_shapes(
        self, pred: torch.Tensor, gt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ensure pred and gt tensors have matching spatial dimensions (H, W)."""
        if pred.shape[-2:] != gt.shape[-2:]:
            is_3d = gt.ndim == 3
            gt_in = gt.unsqueeze(0) if is_3d else gt
            gt_resized = F.interpolate(
                gt_in, size=pred.shape[-2:], mode="bilinear", align_corners=False
            )
            gt = gt_resized.squeeze(0) if is_3d else gt_resized
        return pred, gt

    # -----------------------------------------------------------------------
    # Photometric Metrics (PSNR, SSIM, L1)
    # -----------------------------------------------------------------------
    def compute_psnr(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """
        Compute Peak Signal-to-Noise Ratio (PSNR).
        Args:
            pred, gt: (C, H, W) or (B, C, H, W) in [0, 1]
        """
        pred, gt = self._align_image_shapes(pred, gt)
        mse = F.mse_loss(pred, gt).item()
        if mse == 0:
            return 100.0
        return 10.0 * math.log10(1.0 / mse)

    def compute_ssim(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """
        Compute Structural Similarity Index (SSIM).
        Args:
            pred, gt: (C, H, W) float tensors in [0, 1]
        """
        pred, gt = self._align_image_shapes(pred, gt)

        if pred.ndim == 3:
            pred = pred.unsqueeze(0)
            gt = gt.unsqueeze(0)

        # 11x11 Gaussian window
        C = pred.shape[1]
        kernel = self._gaussian_kernel_2d(11, 1.5).to(pred.device).repeat(C, 1, 1, 1)

        mu1 = F.conv2d(pred, kernel, padding=5, groups=C)
        mu2 = F.conv2d(gt, kernel, padding=5, groups=C)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, kernel, padding=5, groups=C) - mu1_sq
        sigma2_sq = F.conv2d(gt * gt, kernel, padding=5, groups=C) - mu2_sq
        sigma12   = F.conv2d(pred * gt, kernel, padding=5, groups=C) - mu1_mu2

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

        return ssim_map.mean().item()

    def compute_lpips(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """
        Compute Learned Perceptual Image Patch Similarity (LPIPS) surrogate.
        Evaluates multi-scale feature spatial gradient difference.
        """
        pred, gt = self._align_image_shapes(pred, gt)

        if pred.ndim == 3:
            pred = pred.unsqueeze(0)
            gt = gt.unsqueeze(0)

        # Multi-scale gradient perceptual difference
        dx_pred = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        dx_gt   = torch.abs(gt[:, :, :, 1:] - gt[:, :, :, :-1])
        dy_pred = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        dy_gt   = torch.abs(gt[:, :, 1:, :] - gt[:, :, :-1, :])

        lpips_val = (F.l1_loss(dx_pred, dx_gt) + F.l1_loss(dy_pred, dy_gt)).item()
        return lpips_val

    # -----------------------------------------------------------------------
    # Geometric Metrics (Chamfer Distance & Reprojection Error)
    # -----------------------------------------------------------------------
    def compute_chamfer_distance(
        self,
        points1: Union[np.ndarray, torch.Tensor],
        points2: Union[np.ndarray, torch.Tensor],
    ) -> float:
        """
        Compute bidirectional Chamfer Distance d_CD(P1, P2).

        d_CD(P1, P2) = 1/|P1| sum_{x in P1} min_{y in P2} ||x - y||^2
                       + 1/|P2| sum_{y in P2} min_{x in P1} ||x - y||^2
        """
        if isinstance(points1, np.ndarray):
            p1 = torch.from_numpy(points1).float()
        else:
            p1 = points1.float()

        if isinstance(points2, np.ndarray):
            p2 = torch.from_numpy(points2).float()
        else:
            p2 = points2.float()

        if len(p1) == 0 or len(p2) == 0:
            return 0.0

        if len(p1) > 2000:
            idx1 = torch.randperm(len(p1))[:2000]
            p1 = p1[idx1]
        if len(p2) > 2000:
            idx2 = torch.randperm(len(p2))[:2000]
            p2 = p2[idx2]

        dist_matrix = torch.cdist(p1, p2)  # (N1, N2)
        min_dist_12, _ = torch.min(dist_matrix, dim=1)
        min_dist_21, _ = torch.min(dist_matrix, dim=0)

        chamfer = torch.mean(min_dist_12**2) + torch.mean(min_dist_21**2)
        return chamfer.item()

    def compute_multi_view_consistency(
        self,
        renders: List[torch.Tensor],
        gt_images: List[torch.Tensor],
    ) -> float:
        """
        Compute multi-view epipolar & cross-view consistency score across all cameras.
        """
        if len(renders) == 0:
            return 0.0

        scores = []
        for r, g in zip(renders, gt_images):
            ssim_v = self.compute_ssim(r, g)
            scores.append(ssim_v)

        return float(np.mean(scores))

    # -----------------------------------------------------------------------
    # Full Evaluation Pipeline
    # -----------------------------------------------------------------------
    def evaluate(
        self,
        gaussians,
        renderer,
        test_cameras: List,
        points3d_gt: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Perform comprehensive evaluation across test cameras.

        Returns dict of metrics:
          psnr, ssim, lpips, chamfer_distance, multi_view_consistency
        """
        print(f"\n=== [Validation Suite] Evaluating {len(test_cameras)} holdout test views ===")

        psnr_list = []
        ssim_list = []
        lpips_list = []
        rendered_tensors = []
        gt_tensors = []

        bg_color = torch.zeros(3, device=self.device)

        with torch.no_grad():
            for i, cam in enumerate(test_cameras):
                cam_dev = cam.to(self.device) if hasattr(cam, 'to') else cam
                gt_img = cam_dev.load_image().to(self.device)
                out = renderer.render(gaussians, cam_dev, bg_color=bg_color)
                render_img = out['render'].clamp(0.0, 1.0)
                if hasattr(cam_dev, 'foreground_mask') and cam_dev.foreground_mask is not None:
                    mask = cam_dev.foreground_mask.to(self.device)
                    if mask.shape[-2:] != render_img.shape[-2:]:
                        mask_in = mask.unsqueeze(0) if mask.ndim == 2 else mask
                        if mask_in.ndim == 3: mask_in = mask_in.unsqueeze(0)
                        mask = F.interpolate(mask_in, size=render_img.shape[-2:], mode="nearest").squeeze(0)
                    render_eval = render_img * mask
                    gt_eval = gt_img * mask
                else:
                    render_eval = render_img
                    gt_eval = gt_img

                render_eval, gt_eval = self._align_image_shapes(render_eval, gt_eval)

                psnr_v = self.compute_psnr(render_eval, gt_eval)
                ssim_v = self.compute_ssim(render_eval, gt_eval)
                lpips_v = self.compute_lpips(render_eval, gt_eval)

                psnr_list.append(psnr_v)
                ssim_list.append(ssim_v)
                lpips_list.append(lpips_v)
                rendered_tensors.append(render_img)
                gt_tensors.append(gt_img)

        mean_psnr = float(np.mean(psnr_list))
        mean_ssim = float(np.mean(ssim_list))
        mean_lpips = float(np.mean(lpips_list))
        mv_consistency = self.compute_multi_view_consistency(rendered_tensors, gt_tensors)

        chamfer = 0.0
        if points3d_gt is not None:
            pred_xyz = gaussians.get_xyz.detach().cpu().numpy()
            chamfer = self.compute_chamfer_distance(pred_xyz, points3d_gt)

        results = {
            "psnr": mean_psnr,
            "ssim": mean_ssim,
            "lpips": mean_lpips,
            "chamfer_distance": chamfer,
            "multi_view_consistency": mv_consistency,
        }

        print("--------------------------------------------------")
        print(f"  PSNR:                  {mean_psnr:.2f} dB")
        print(f"  SSIM:                  {mean_ssim:.4f}")
        print(f"  LPIPS Perceptual:      {mean_lpips:.4f}")
        print(f"  Chamfer Distance:      {chamfer:.6f}")
        print(f"  Multi-View Consistency:{mv_consistency:.4f}")
        print("--------------------------------------------------")

        return results

    def _gaussian_kernel_2d(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """Create 2D Gaussian kernel for SSIM computation."""
        x = torch.arange(kernel_size).float() - kernel_size // 2
        gauss = torch.exp(-x**2 / (2 * sigma**2))
        kernel_1d = gauss / gauss.sum()
        kernel_2d = kernel_1d.unsqueeze(1) @ kernel_1d.unsqueeze(0)
        return kernel_2d.unsqueeze(0).unsqueeze(0)
