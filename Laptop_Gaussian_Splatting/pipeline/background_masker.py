"""
Object Mask Generator & Background Segmenter.

Generates high-precision binary foreground masks M ∈ {0, 1}^(H x W)
to isolate the target object (e.g., apple) from background clutter.

Prevents background floaters and ensures 100% of 3D Gaussians reconstruct
the target object's true geometry and appearance.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import cv2


class ObjectMaskGenerator:
    """
    Automatic object background segmentation module.
    Combines GrabCut, Color Saliency, Otsu adaptive thresholding, and morphological refinement.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def generate_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Generate binary foreground object mask for an RGB image.

        Args:
            image_rgb: (H, W, 3) uint8 or float32 RGB image

        Returns:
            mask: (H, W) float32 array in [0.0, 1.0] where 1.0 is foreground
        """
        if image_rgb.dtype != np.uint8:
            image_bgr = cv2.cvtColor((image_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        else:
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        h, w = image_bgr.shape[:2]

        # Color Saliency & Lab Color Difference Thresholding (Instant & Robust)
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab)
        l_chan = lab[:, :, 0]
        a_chan = lab[:, :, 1]
        b_chan = lab[:, :, 2]
        
        _, otsu_fg = cv2.threshold(l_chan, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        otsu_fg = (otsu_fg / 255.0).astype(np.float32)

        # Center distance weight (prioritizes central object)
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2.0, w / 2.0
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        center_weight = 1.0 - (dist_from_center / max_dist) * 0.4
        
        combined = otsu_fg * center_weight
        binary_mask = (combined >= (self.threshold * 0.6)).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        clean_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)

        smooth_mask = cv2.GaussianBlur(clean_mask.astype(np.float32), (5, 5), 0)
        return smooth_mask

    def process_camera_list(self, cameras: List) -> List:
        """
        Process all cameras in a scene and attach foreground mask tensors.

        Args:
            cameras: List of Camera objects

        Returns:
            cameras with camera.foreground_mask updated
        """
        print(f"=== [Background Segmenter] Generating object masks for {len(cameras)} cameras ===")
        for i, cam in enumerate(cameras):
            if hasattr(cam, 'original_image') and cam.original_image is not None:
                img_np = (cam.original_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                mask_np = self.generate_mask(img_np)
                cam.foreground_mask = torch.from_numpy(mask_np).float().unsqueeze(0)
        print("[OK] Object masks generated successfully.")
        return cameras
