"""
[EXPERIMENTAL] Neural Quality Balancer — MLP-guided densification.

A small fully-connected network predicts per-Gaussian importance scores
based on learned features. High-score Gaussians are densified; low-score
ones are pruned. This replaces/augments the heuristic gradient-based
criterion with a learned decision.

Input features per Gaussian:
  1. opacity           (sigmoid activated)
  2. mean scale        (mean of exp(log_scale))
  3. SH DC magnitude   (L2 norm of DC color vector)
  4. gradient norm     (accumulated |∇_xy|² from densification stats)

Output:
  score ∈ [0, 1]    (1 = densify, 0 = prune)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class NeuralQualityBalancer(nn.Module):
    """
    3-layer MLP predicting per-Gaussian importance for adaptive densification.

    Architecture:
        input (4) → Linear → ReLU → Linear → ReLU → Linear → Sigmoid → output (1)
        hidden dim = 64
    """

    def __init__(self, input_dim: int = 4, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
        # Initialize final layer near 0.5 (balanced prior)
        nn.init.zeros_(self.net[-2].weight)
        nn.init.constant_(self.net[-2].bias, 0.0)

    # -----------------------------------------------------------------------
    # Feature extraction
    # -----------------------------------------------------------------------

    @staticmethod
    def extract_features(gaussians, gradient_accum: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract per-Gaussian features for importance prediction.

        Args:
            gaussians:       GaussianModel instance
            gradient_accum:  (N, 1) accumulated 2D gradient norms

        Returns:
            (N, 4) feature tensor [opacity, mean_scale, sh_dc_mag, grad_norm]
        """
        N = gaussians.num_gaussians
        device = gaussians.get_xyz.device

        # Feature 1: opacity ∈ [0, 1]
        opacity = gaussians.get_opacity.squeeze(-1)  # (N,)

        # Feature 2: mean scale (geometric mean of x,y,z scales)
        mean_scale = gaussians.get_scaling.mean(dim=-1)  # (N,)

        # Feature 3: SH DC color magnitude (L2 norm of 3 DC channels)
        sh_dc = gaussians._features_dc.squeeze(1)  # (N, 3)
        sh_mag = sh_dc.norm(dim=-1)  # (N,)

        # Feature 4: accumulated gradient norm
        if gradient_accum is not None:
            avg_grad = (gradient_accum / gradient_accum.clamp(min=1)).squeeze(-1)
        else:
            avg_grad = torch.zeros(N, device=device)

        # Stack and normalize each feature to [0, 1]
        feats = torch.stack([opacity, mean_scale, sh_mag, avg_grad], dim=-1)  # (N, 4)
        feats = (feats - feats.min(dim=0).values) / (
            feats.max(dim=0).values - feats.min(dim=0).values + 1e-8
        )
        return feats

    # -----------------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------------

    def predict(self, gaussians, gradient_accum: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict importance scores for all Gaussians.

        Args:
            gaussians:      GaussianModel
            gradient_accum: (N, 1) accumulated gradient norms

        Returns:
            (N,) scores in [0, 1]
        """
        feats = self.extract_features(gaussians, gradient_accum)
        with torch.no_grad():
            scores = self.net(feats).squeeze(-1)
        return scores

    # -----------------------------------------------------------------------
    # Training step
    # -----------------------------------------------------------------------

    def train_step(
        self,
        gaussians,
        rendered_psnr_delta: float,
        gradient_accum: Optional[torch.Tensor] = None,
        lr: float = 1e-3,
    ) -> float:
        """
        Supervised training step: predict scores, compute loss based on
        whether PSNR improved after recent densification.

        Targets: if psnr_delta > 0, gaussians with high gradient are good candidates.

        Args:
            gaussians:          GaussianModel
            rendered_psnr_delta: change in PSNR from last densification (+ve = improved)
            gradient_accum:     (N, 1) accumulated gradient norms
            lr:                 learning rate for this step

        Returns:
            scalar loss value
        """
        feats = self.extract_features(gaussians, gradient_accum)

        # Pseudo-label: use gradient norm as target (high gradient = should densify)
        if gradient_accum is not None:
            g = gradient_accum.squeeze(-1).float()
            target = (g / (g.max() + 1e-8)).clamp(0.0, 1.0)
        else:
            target = torch.zeros(gaussians.num_gaussians, device=feats.device)

        # Adjust targets by PSNR signal: if PSNR dropped, be more conservative
        if rendered_psnr_delta < 0:
            target = target * 0.5  # reduce densification aggressiveness

        scores = self.net(feats).squeeze(-1)
        loss = F.binary_cross_entropy(scores, target.detach())

        # Manual gradient step
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()
