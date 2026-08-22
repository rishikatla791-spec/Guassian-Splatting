"""
Evaluation Metrics & Test View Rendering Benchmark.
Computes PSNR, SSIM, and saves test-view reconstruction comparisons.
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image as PILImage
import torchvision.transforms.functional as TF

from core.gaussians import GaussianModel
from core.camera import Camera
from renderer.tile_rasterizer import TileBasedRasterizer
from training.loss import ssim_loss, l1_loss


def compute_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) in dB.
    Args:
        pred: (3, H, W) float32 tensor in [0, 1]
        gt:   (3, H, W) float32 tensor in [0, 1]
    """
    mse = F.mse_loss(pred, gt).item()
    if mse <= 1e-10:
        return 100.0
    return float(-10.0 * np.log10(mse))


def compute_ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    Compute Structural Similarity Index (SSIM) in [0, 1].
    Args:
        pred: (3, H, W) float32 tensor in [0, 1]
        gt:   (3, H, W) float32 tensor in [0, 1]
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)
    loss = ssim_loss(pred, gt)
    return float(1.0 - loss.item())


_LPIPS_MODEL = None

def get_lpips_model(device: str = "cuda"):
    global _LPIPS_MODEL
    if _LPIPS_MODEL is None:
        try:
            import lpips
            _LPIPS_MODEL = lpips.LPIPS(net='vgg', verbose=False).to(device)
            _LPIPS_MODEL.eval()
        except Exception as e:
            print(f"[Warning] Failed to initialize LPIPS: {e}")
            _LPIPS_MODEL = None
    return _LPIPS_MODEL


def compute_lpips(pred: torch.Tensor, gt: torch.Tensor, lpips_model=None) -> float:
    """
    Compute Learned Perceptual Image Patch Similarity (LPIPS).
    Args:
        pred: (3, H, W) float32 tensor in [0, 1]
        gt:   (3, H, W) float32 tensor in [0, 1]
    """
    if lpips_model is None:
        lpips_model = get_lpips_model(device=pred.device)
    if lpips_model is None:
        return 0.0

    # LPIPS expects input normalized to [-1, 1] with shape (1, 3, H, W)
    p = (pred.unsqueeze(0) * 2.0 - 1.0).clamp(-1.0, 1.0)
    g = (gt.unsqueeze(0) * 2.0 - 1.0).clamp(-1.0, 1.0)
    with torch.no_grad():
        val = lpips_model(p, g).item()
    return float(val)


def evaluate_dataset(
    gaussians: GaussianModel,
    test_cameras: List[Camera],
    renderer: TileBasedRasterizer,
    save_dir: Optional[str | Path] = None,
    bg_color: Optional[torch.Tensor] = None,
    compute_lpips_metric: bool = True,
) -> Dict[str, float]:
    """
    Evaluate trained Gaussian model on test cameras.

    Returns:
        dict with 'mean_psnr', 'mean_ssim', 'mean_lpips', 'mean_l1', 'num_views'
    """
    device = gaussians.get_xyz.device
    if bg_color is None:
        bg_color = torch.zeros(3, device=device)

    lpips_fn = get_lpips_model(device=device) if compute_lpips_metric else None

    if save_dir:
        save_dir = Path(save_dir)
        render_dir = save_dir / "renders"
        gt_dir = save_dir / "gt"
        render_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)

    psnr_list = []
    ssim_list = []
    lpips_list = []
    l1_list = []

    print(f"\n[Evaluation] Evaluating on {len(test_cameras)} unseen test views...")

    with torch.no_grad():
        for idx, cam in enumerate(test_cameras):
            out = renderer.render(gaussians, cam, bg_color=bg_color)
            rendered = out["render"].clamp(0.0, 1.0) # (3, H, W)

            gt_img = cam.load_image().to(device)
            if gt_img.shape[-2:] != rendered.shape[-2:]:
                gt_img = F.interpolate(
                    gt_img.unsqueeze(0),
                    size=rendered.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)

            psnr_val = compute_psnr(rendered, gt_img)
            ssim_val = compute_ssim(rendered, gt_img)
            l1_val = l1_loss(rendered, gt_img).item()

            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
            l1_list.append(l1_val)

            if lpips_fn is not None:
                lp_val = compute_lpips(rendered, gt_img, lpips_model=lpips_fn)
                lpips_list.append(lp_val)

            if save_dir:
                # Save rendered PNG
                rend_np = (rendered.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                gt_np = (gt_img.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

                cam_name = Path(cam.image_path).stem if cam.image_path else f"view_{idx:04d}"
                PILImage.fromarray(rend_np).save(render_dir / f"{cam_name}.png")
                PILImage.fromarray(gt_np).save(gt_dir / f"{cam_name}.png")

    mean_psnr = float(np.mean(psnr_list))
    mean_ssim = float(np.mean(ssim_list))
    mean_l1 = float(np.mean(l1_list))
    mean_lpips = float(np.mean(lpips_list)) if lpips_list else 0.0

    print(f"[Evaluation Summary] Mean PSNR: {mean_psnr:.2f} dB | Mean SSIM: {mean_ssim:.4f} | Mean LPIPS: {mean_lpips:.4f} | Mean L1: {mean_l1:.4f}")

    return {
        "mean_psnr": mean_psnr,
        "mean_ssim": mean_ssim,
        "mean_lpips": mean_lpips,
        "mean_l1": mean_l1,
        "num_views": len(test_cameras),
        "psnr_list": psnr_list,
        "ssim_list": ssim_list,
        "lpips_list": lpips_list,
    }
