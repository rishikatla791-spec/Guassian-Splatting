"""
[EXPERIMENTAL] Semantic Awareness Module for 3D Gaussian Splatting.

Equips 3D Gaussians with learnable per-Gaussian semantic features/class logits.
Enables semantic rendering, class-based object extraction/filtering, and semantic-guided densification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class SemanticGaussianExtension:
    """
    Extends a GaussianModel with per-Gaussian semantic feature embeddings or class logits.
    """

    def __init__(self, num_classes: int = 8, feature_dim: int = 16):
        self.num_classes = num_classes
        self.feature_dim = feature_dim

    def attach_semantics(self, gaussians, initial_classes: Optional[torch.Tensor] = None) -> nn.Parameter:
        """
        Attach learnable semantic logits to a GaussianModel.

        Args:
            gaussians: GaussianModel instance
            initial_classes: optional (N,) class labels for initialization

        Returns:
            semantics_param: (N, num_classes) nn.Parameter
        """
        N = gaussians.num_gaussians
        device = gaussians.get_xyz.device

        if initial_classes is not None:
            one_hot = F.one_hot(initial_classes.long(), num_classes=self.num_classes).float()
            logits = torch.log(one_hot.clamp(min=1e-3, max=1.0 - 1e-3))
        else:
            logits = torch.zeros(N, self.num_classes, device=device)

        semantics_param = nn.Parameter(logits)
        gaussians.semantics = semantics_param
        return semantics_param

    def get_semantic_probabilities(self, gaussians) -> torch.Tensor:
        """Return (N, num_classes) softmax class probabilities."""
        if not hasattr(gaussians, "semantics"):
            raise AttributeError("Semantics not attached to GaussianModel. Call attach_semantics first.")
        return F.softmax(gaussians.semantics, dim=-1)

    def render_semantic_map(
        self,
        gaussians,
        camera,
        rasterizer,
        class_colors: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Render a 2D semantic class map from current camera view.

        Args:
            gaussians: GaussianModel with attached semantics
            camera: Camera instance
            rasterizer: TileBasedRasterizer
            class_colors: (num_classes, 3) RGB palette for visualization

        Returns:
            Dict containing 'semantic_image' (3, H, W) and 'class_probabilities' (num_classes, H, W)
        """
        probs = self.get_semantic_probabilities(gaussians)  # (N, num_classes)
        pred_class = probs.argmax(dim=-1)                   # (N,)

        if class_colors is None:
            # Default distinct colormap
            torch.manual_seed(123)
            class_colors = torch.rand(self.num_classes, 3, device=probs.device)

        # Map classes to RGB colors for each Gaussian
        gaussian_colors = class_colors[pred_class]  # (N, 3)

        bg = torch.zeros(3, device=probs.device)
        out = rasterizer.render(gaussians, camera, bg_color=bg, override_colors=gaussian_colors)

        return {
            "semantic_image": out["render"],
            "pred_classes": pred_class,
            "probabilities": probs,
        }

    def filter_by_class(self, gaussians, target_class: int, threshold: float = 0.5) -> torch.Tensor:
        """
        Create boolean mask selecting Gaussians belonging to target_class.

        Args:
            gaussians: GaussianModel
            target_class: target class ID in [0, num_classes)
            threshold: probability threshold

        Returns:
            (N,) bool mask
        """
        probs = self.get_semantic_probabilities(gaussians)
        return probs[:, target_class] >= threshold
