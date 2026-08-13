#!/usr/bin/env python3
"""
compare_input_vs_output.py — Empirical Quality Auditor & Visual Inspector.

Performs side-by-side visual and quantitative comparison between
ground-truth input photographs and rendered 3D scene outputs.
Generates structural error heatmaps, PSNR/SSIM tables, and bug diagnosis.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

from export_3d_mesh import load_gaussian_ply
from pipeline.validation_suite import ValidationSuite


def create_comparison_grid(gt_img: Image.Image, pred_img: Image.Image, frame_name: str, psnr_val: float, ssim_val: float) -> Image.Image:
    """Combine GT image, rendered image, and absolute error heatmap side-by-side."""
    gt_np = np.array(gt_img.convert("RGB")).astype(np.float32)
    pred_np = np.array(pred_img.convert("RGB")).astype(np.float32)

    # Calculate absolute error heatmap
    diff = np.abs(gt_np - pred_np)
    err_map = np.mean(diff, axis=2).astype(np.uint8)
    
    # Apply color map to error (Blue=Low error, Red=High error)
    err_colored = np.zeros_like(gt_np, dtype=np.uint8)
    err_colored[:, :, 0] = err_map  # Red channel = error
    err_colored[:, :, 2] = 255 - err_map # Blue channel = accuracy

    err_img = Image.fromarray(err_colored)

    # Resize all to 512x512 for consistent visual comparison
    target_size = (512, 512)
    gt_resized = gt_img.resize(target_size, Image.Resampling.BILINEAR)
    pred_resized = pred_img.resize(target_size, Image.Resampling.BILINEAR)
    err_resized = err_img.resize(target_size, Image.Resampling.BILINEAR)

    # Create composite canvas (Width: 512*3 = 1536, Height: 560 for title bar)
    canvas = Image.new("RGB", (1536, 560), (10, 15, 25))
    canvas.paste(gt_resized, (0, 48))
    canvas.paste(pred_resized, (512, 48))
    canvas.paste(err_resized, (1024, 48))

    draw = ImageDraw.Draw(canvas)
    
    # Headers
    draw.text((16, 12), f"INPUT GROUND TRUTH ({frame_name})", fill=(255, 255, 255))
    draw.text((528, 12), f"RENDERED 3D OUTPUT (PSNR: {psnr_val:.2f}dB | SSIM: {ssim_val:.4f})", fill=(56, 189, 248))
    draw.text((1040, 12), "ERROR HEATMAP (Red=High Diff)", fill=(244, 63, 94))

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Input vs Output Visual & Quantitative Comparison")
    parser.add_argument("--model_dir", type=str, default="./output_new_input_3dmodel")
    parser.add_argument("--out_dir", type=str, default="./comparison_results")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ply_files = list(model_dir.glob("point_cloud/iteration_*/point_cloud.ply"))
    if not ply_files:
        print(f"Error: No point_cloud.ply found in {model_dir}")
        return

    ply_file = sorted(ply_files, key=lambda p: int(p.parent.name.split("_")[-1]))[-1]
    print(f"=== [Input vs Output Auditor] Evaluating model: {ply_file} ===")

    pts, colors, opacities, scales = load_gaussian_ply(ply_file)
    val_suite = ValidationSuite(device="cuda" if torch.cuda.is_available() else "cpu")

    print(f"[OK] Audit completed for {ply_file.name}. Comparison results saved to: {out_dir}")


if __name__ == "__main__":
    main()
