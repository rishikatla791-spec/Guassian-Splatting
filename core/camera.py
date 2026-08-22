"""
Camera model for 3D Gaussian Splatting.

Supports pinhole camera with full intrinsic/extrinsic parametrization.
Follows OpenCV (right-handed) world coordinate convention with OpenGL-style
projection matrices for rendering.

Coordinate conventions:
  World:  right-handed (x right, y up, z out of screen) -- COLMAP convention
  Camera: right-handed (x right, y down, z into scene)  -- OpenCV convention
  NDC:    [-1,1]³ with z=−1 near, z=+1 far (OpenGL)

View matrix:   W→C transform: [R|t] (4×4)
Proj matrix:   C→clip:  OpenGL frustum (4×4)
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsics."""
    fx: float                  # focal length x (pixels)
    fy: float                  # focal length y (pixels)
    cx: float                  # principal point x (pixels)
    cy: float                  # principal point y (pixels)
    width: int                 # image width (pixels)
    height: int                # image height (pixels)

    @property
    def fovx(self) -> float:
        """Horizontal field of view in radians."""
        return 2.0 * math.atan(self.width / (2.0 * self.fx))

    @property
    def fovy(self) -> float:
        """Vertical field of view in radians."""
        return 2.0 * math.atan(self.height / (2.0 * self.fy))

    @property
    def K(self) -> np.ndarray:
        """3×3 intrinsic matrix."""
        return np.array([
            [self.fx,       0, self.cx],
            [      0, self.fy, self.cy],
            [      0,       0,       1],
        ], dtype=np.float64)

    @classmethod
    def from_fov(cls, fovx: float, fovy: float, width: int, height: int) -> "CameraIntrinsics":
        fx = width / (2.0 * math.tan(fovx / 2.0))
        fy = height / (2.0 * math.tan(fovy / 2.0))
        return cls(fx=fx, fy=fy, cx=width / 2.0, cy=height / 2.0, width=width, height=height)


@dataclass
class CameraExtrinsics:
    """Camera extrinsics: world → camera transform."""
    R: np.ndarray   # (3, 3) rotation matrix (world → camera)
    T: np.ndarray   # (3,) translation (world → camera)

    @property
    def camera_center(self) -> np.ndarray:
        """Camera center in world space: C = -Rᵀ T."""
        return -self.R.T @ self.T

    @property
    def view_matrix_np(self) -> np.ndarray:
        """4×4 world-to-camera matrix (column-major for OpenGL)."""
        W = np.eye(4, dtype=np.float64)
        W[:3, :3] = self.R
        W[:3, 3] = self.T
        return W

    @classmethod
    def from_look_at(
        cls,
        eye: np.ndarray,
        target: np.ndarray,
        up: np.ndarray = np.array([0.0, 1.0, 0.0]),
    ) -> "CameraExtrinsics":
        """Create extrinsics from look-at parameters."""
        z = eye - target
        z = z / np.linalg.norm(z)
        x = np.cross(up, z)
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        R = np.stack([x, y, z], axis=0)  # rows are camera axes
        T = -R @ eye
        return cls(R=R, T=T)


@dataclass
class Camera:
    """
    Full camera model combining intrinsics and extrinsics.
    Provides all matrices needed for rendering in both PyTorch and NumPy.
    """
    uid: int
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
    image_path: Optional[str] = None
    image: Optional[torch.Tensor] = None   # (3, H, W) float32 [0,1]
    near: float = 0.01
    far: float = 100.0

    # -----------------------------------------------------------------------
    # View matrix
    # -----------------------------------------------------------------------

    @property
    def view_matrix(self) -> torch.Tensor:
        """(4,4) float32 world-to-camera on CPU."""
        return torch.tensor(self.extrinsics.view_matrix_np, dtype=torch.float32)

    # -----------------------------------------------------------------------
    # Projection matrix (OpenGL-style frustum)
    # -----------------------------------------------------------------------

    def projection_matrix(
        self,
        znear: Optional[float] = None,
        zfar: Optional[float] = None,
    ) -> torch.Tensor:
        """
        4×4 OpenGL-style projection matrix mapping camera space → clip space.

        For a pinhole camera with (cx, cy) at the image center:

            P = [[2fx/W,     0,  (W-2cx)/W,         0      ],
                 [    0,  2fy/H, (H-2cy)/H,         0      ],
                 [    0,     0,  -(f+n)/(f-n), -2fn/(f-n)  ],
                 [    0,     0,        -1,           0      ]]

        Args:
            znear: near clip plane (default self.near)
            zfar:  far clip plane  (default self.far)

        Returns:
            (4, 4) float32 projection matrix
        """
        n = znear if znear is not None else self.near
        f = zfar if zfar is not None else self.far
        K = self.intrinsics
        W, H = K.width, K.height

        P = torch.zeros(4, 4, dtype=torch.float32)
        P[0, 0] = 2.0 * K.fx / W
        P[0, 2] = (W - 2.0 * K.cx) / W
        P[1, 1] = 2.0 * K.fy / H
        P[1, 2] = (H - 2.0 * K.cy) / H
        P[2, 2] = (f + n) / (f - n)
        P[2, 3] = -2.0 * f * n / (f - n)
        P[3, 2] = 1.0

        return P

    @property
    def full_proj_transform(self) -> torch.Tensor:
        """(4,4) combined projection × view matrix."""
        return self.projection_matrix() @ self.view_matrix

    @property
    def camera_center(self) -> torch.Tensor:
        """(3,) camera center in world space."""
        return torch.tensor(self.extrinsics.camera_center, dtype=torch.float32)

    # -----------------------------------------------------------------------
    # Convenience properties
    # -----------------------------------------------------------------------

    @property
    def width(self) -> int:
        return self.intrinsics.width

    @property
    def height(self) -> int:
        return self.intrinsics.height

    @property
    def fovx(self) -> float:
        return self.intrinsics.fovx

    @property
    def fovy(self) -> float:
        return self.intrinsics.fovy

    def load_image(self, max_dim: int = 256) -> torch.Tensor:
        """Load and cache image as (3, H, W) float32 tensor, updating intrinsics for ultra-fast training."""
        if self.image is not None:
            return self.image
        if self.image_path is None:
            self.image = torch.zeros((3, self.intrinsics.height, self.intrinsics.width), dtype=torch.float32)
            return self.image
        from PIL import Image as PILImage
        import torchvision.transforms.functional as TF
        img = PILImage.open(self.image_path).convert("RGB")
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            img = img.resize((new_w, new_h), PILImage.BILINEAR)
            # Update camera intrinsics to match resized image footprint
            self.intrinsics.fx *= scale
            self.intrinsics.fy *= scale
            self.intrinsics.cx *= scale
            self.intrinsics.cy *= scale
            self.intrinsics.width = new_w
            self.intrinsics.height = new_h
        self.image = TF.to_tensor(img)  # (3, H, W), [0, 1]
        return self.image

    def to(self, device: torch.device | str) -> "Camera":
        """Move tensors to device."""
        if self.image is not None:
            self.image = self.image.to(device)
        return self

    def __repr__(self) -> str:
        return (f"Camera(uid={self.uid}, "
                f"size={self.width}×{self.height}, "
                f"fovx={math.degrees(self.fovx):.1f}°, "
                f"path={self.image_path})")


def build_proj_matrix(fovx: float, fovy: float, znear: float, zfar: float) -> torch.Tensor:
    """
    Build OpenGL-style projection matrix from FoV parameters.
    Used when intrinsics are parametrized by FoV only.

    tan(fovx/2) = W/(2fx), similarly for y.
    """
    tan_half_x = math.tan(fovx / 2.0)
    tan_half_y = math.tan(fovy / 2.0)

    P = torch.zeros(4, 4, dtype=torch.float32)
    P[0, 0] = 1.0 / tan_half_x
    P[1, 1] = 1.0 / tan_half_y
    P[2, 2] = (zfar + znear) / (zfar - znear)
    P[2, 3] = -2.0 * zfar * znear / (zfar - znear)
    P[3, 2] = 1.0

    return P
