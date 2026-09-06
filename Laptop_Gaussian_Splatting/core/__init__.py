"""
Core module for Next-Generation 3D Gaussian Splatting.
Contains mathematical primitives, CUDA kernels, and core data structures.
"""
from .gaussians import GaussianModel
from .camera import Camera, CameraIntrinsics, CameraExtrinsics
from .math_utils import (
    build_covariance_3d,
    build_covariance_2d,
    compute_sh_coefficients,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    strip_lowerdiag,
    strip_symmetric,
)
from .sh import eval_sh, RGB2SH, SH2RGB, num_sh_coefficients

__all__ = [
    "GaussianModel",
    "Camera",
    "CameraIntrinsics",
    "CameraExtrinsics",
    "build_covariance_3d",
    "build_covariance_2d",
    "compute_sh_coefficients",
    "quaternion_to_rotation_matrix",
    "rotation_matrix_to_quaternion",
    "strip_lowerdiag",
    "strip_symmetric",
    "eval_sh",
    "RGB2SH",
    "SH2RGB",
    "num_sh_coefficients",
]
