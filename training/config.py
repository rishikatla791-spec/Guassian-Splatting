"""
TrainingConfig: Complete configuration for 3D Gaussian Splatting training.

All hyperparameters are documented with mathematical justification.
Defaults reproduce the original Kerbl et al. 2023 3DGS results.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class TrainingConfig:
    """
    Complete 3DGS training configuration.

    Defaults reproduce Kerbl et al. 2023 results on the standard
    Tanks and Temples, Mip-NeRF 360, and Deep Blending datasets.

    Learning Rate Rationale
    ───────────────────────
    Position LR is highest and decays exponentially:
      position_lr decays 100× over 30k steps (from 1.6e-4 to 1.6e-6).
    Feature LR is moderate; rest features use 1/20 of DC to prevent
      high-frequency SH from dominating early training.
    Opacity LR is high to allow rapid transparency changes.
    Rotation/scale LRs are small: these parameters are already constrained
      (unit quaternion, positive scale).

    Densification Rationale
    ───────────────────────
    densify_from_iter=500: allow initial fit before densifying.
    densify_until_iter=15000: stop densifying at 50% of training.
    densification_interval=100: frequent enough to track appearance of
      under-reconstructed regions, rare enough not to waste time.
    densify_grad_threshold=0.0002: Gaussians with average 2D gradient
      magnitude > this are split/cloned (about 20% of dense scenes).
    """

    # ────────────────────────────────────────────────────
    # Training schedule
    # ────────────────────────────────────────────────────
    iterations: int = 30_000
    """Total training iterations."""

    # ────────────────────────────────────────────────────
    # Position (xyz) learning rate schedule
    # lr(t) = lr_init · (lr_final/lr_init)^(t/T)
    # ────────────────────────────────────────────────────
    position_lr_init: float = 0.00016
    """Initial position learning rate. Corresponds to ~0.016% of a unit scene."""

    position_lr_final: float = 0.0000016
    """Final position learning rate (100× smaller than init)."""

    position_lr_delay_mult: float = 0.01
    """Position LR multiplier at step 0 during sine warmup."""

    position_lr_max_steps: int = 30_000
    """Steps over which to decay position LR."""

    # ────────────────────────────────────────────────────
    # Other parameter learning rates (constant)
    # ────────────────────────────────────────────────────
    feature_lr: float = 0.0025
    """Learning rate for DC SH color features."""

    opacity_lr: float = 0.05
    """Learning rate for logit-opacities. High to allow rapid transparency changes."""

    scaling_lr: float = 0.005
    """Learning rate for log-scale parameters."""

    rotation_lr: float = 0.001
    """Learning rate for rotation quaternions."""

    # ────────────────────────────────────────────────────
    # Loss weights
    # ────────────────────────────────────────────────────
    lambda_dssim: float = 0.2
    """DSSIM weight in combined loss: L = (1-λ)·L1 + λ·DSSIM."""

    # ────────────────────────────────────────────────────
    # Densification
    # ────────────────────────────────────────────────────
    densify_from_iter: int = 500
    """Start densifying after this many iterations (warm-up phase)."""

    densify_until_iter: int = 15_000
    """Stop densifying after this iteration (half of total training)."""

    densification_interval: int = 100
    """Densify every N iterations within [densify_from_iter, densify_until_iter]."""

    opacity_reset_interval: int = 3_000
    """Reset opacities every N iterations to eliminate floaters."""

    densify_grad_threshold: float = 0.0002
    """Average 2D gradient magnitude threshold for split/clone."""

    percent_dense: float = 0.01
    """Percent of scene extent above which Gaussians are split rather than cloned."""

    min_opacity: float = 0.005
    """Prune Gaussians with sigmoid(opacity) < min_opacity."""

    max_screen_size: float = 20.0
    """Prune Gaussians with 2D bounding radius > this (pixels)."""

    # ────────────────────────────────────────────────────
    # Spherical Harmonics
    # ────────────────────────────────────────────────────
    sh_degree: int = 3
    """Maximum SH degree (0–3). Degree increases progressively during training."""

    # ────────────────────────────────────────────────────
    # Renderer flags
    # ────────────────────────────────────────────────────
    convert_SHs_python: bool = False
    """Evaluate SH in Python (True) vs. in CUDA kernel (False)."""

    compute_cov3D_python: bool = False
    """Compute 3D covariance in Python (True) vs. in CUDA kernel (False)."""

    white_background: bool = False
    """Composite against white background."""

    random_background: bool = True
    """Use random background color during training (improves generalization)."""

    # ────────────────────────────────────────────────────
    # Output & checkpointing
    # ────────────────────────────────────────────────────
    model_path: str = './output'
    """Root directory for all training outputs."""

    save_iterations: List[int] = field(
        default_factory=lambda: [7_000, 30_000]
    )
    """Iterations at which to save full PLY model."""

    test_iterations: List[int] = field(
        default_factory=lambda: [7_000, 30_000]
    )
    """Iterations at which to evaluate test-set metrics."""

    checkpoint_iterations: List[int] = field(
        default_factory=list
    )
    """Iterations at which to save full training checkpoints (with optimizer state)."""

    quiet: bool = False
    """Suppress per-iteration progress output."""

    # ────────────────────────────────────────────────────
    # Data loading
    # ────────────────────────────────────────────────────
    resolution: int = -1
    """Image resolution override (-1 = original)."""

    data_device: str = 'cuda'
    """Device for training images ('cuda' or 'cpu')."""

    # ────────────────────────────────────────────────────
    # Serialization
    # ────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Convert to plain Python dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingConfig":
        """
        Create TrainingConfig from dict.
        Unknown keys are silently ignored for forward/backward compatibility.
        """
        valid_fields = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        list_fields = {'save_iterations', 'test_iterations', 'checkpoint_iterations'}
        for lf in list_fields:
            if lf in filtered and not isinstance(filtered[lf], list):
                filtered[lf] = list(filtered[lf])
        return cls(**filtered)

    def save_json(self, path: str) -> None:
        """Save config to JSON file, creating parent dirs as needed."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load_json(cls, path: str) -> "TrainingConfig":
        """Load config from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def validate(self) -> None:
        """
        Validate configuration values and raise informative errors.

        Checks:
            - SH degree in [0, 3]
            - LR values positive
            - Densification window is valid
            - max_screen_size positive
        """
        assert 0 <= self.sh_degree <= 3, f"sh_degree must be 0-3, got {self.sh_degree}"
        assert self.position_lr_init > 0, "position_lr_init must be positive"
        assert self.position_lr_final > 0, "position_lr_final must be positive"
        assert self.feature_lr > 0, "feature_lr must be positive"
        assert self.opacity_lr > 0, "opacity_lr must be positive"
        assert 0.0 < self.lambda_dssim < 1.0, "lambda_dssim must be in (0,1)"
        assert self.densify_from_iter < self.densify_until_iter, (
            "densify_from_iter must be < densify_until_iter"
        )
        assert self.min_opacity > 0, "min_opacity must be positive"
        assert self.max_screen_size > 0, "max_screen_size must be positive"
        assert self.iterations > 0, "iterations must be positive"

    def __repr__(self) -> str:
        fields = "\n  ".join(f"{k}={v!r}" for k, v in self.to_dict().items())
        return f"TrainingConfig(\n  {fields}\n)"
