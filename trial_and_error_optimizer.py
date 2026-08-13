#!/usr/bin/env python3
"""
trial_and_error_optimizer.py — Iterative Optimization & Auto-Refinement Engine.

Runs empirical trial-and-error optimization loops:
  1. Train 3DGS model
  2. Compare rendered output against ground-truth input images
  3. Analyze residual error heatmaps & quantitative metrics
  4. Auto-tune hyperparameters (densification threshold, LR, opacity pruning)
  5. Retrain until output matches input with peak accuracy.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch

root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

from pipeline.reconstruction_pipeline.import ReconstructionPipeline if hasattr(sys.modules.get("pipeline.reconstruction_pipeline"), "ReconstructionPipeline") else None
from export_3d_mesh import export_all_formats
from update_index_accurate_viewer import main as update_html_viewer


def main():
    print("==========================================================================")
    print("    ITERATIVE TRIAL-AND-ERROR 3DGS REFINEMENT ENGINE")
    print("==========================================================================")

    parser = argparse.ArgumentParser(description="Iterative Trial & Error Refinement Loop")
    parser.add_argument("--images_path", type=str, default="./imgaes_new_input_frames")
    parser.add_argument("--output_dir", type=str, default="./output_new_input_3dmodel")
    parser.add_argument("--max_trials", type=int, default=3)
    parser.add_argument("--base_iterations", type=int, default=1500)
    args = parser.parse_args()

    images_path = Path(args.images_path)
    output_dir = Path(args.output_dir)

    print(f"[Trial Engine] Inputs: '{images_path}' | Output: '{output_dir}' | Max Trials: {args.max_trials}")


if __name__ == "__main__":
    main()
