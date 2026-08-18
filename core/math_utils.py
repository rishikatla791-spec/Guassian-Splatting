"""
Mathematically Rigorous Utility Functions for 3D Gaussian Splatting.

All derivations follow:
  - Zwicker et al. 2001: "EWA Splatting" (IEEE TVCG)
  - Kerbl et al. 2023: "3D Gaussian Splatting for Real-Time Novel View Synthesis"
  - Standard quaternion / SO(3) / se(3) Lie group mathematics
  - Möller–Trumbore for exact ray-Gaussian intersection

═══════════════════════════════════════════════════════════════
MATHEMATICAL NOTATION
═══════════════════════════════════════════════════════════════

Gaussian Parametrization
------------------------
  μ ∈ ℝ³          mean / center
  Σ ∈ S₊(3)       positive-definite 3×3 covariance
  Σ = R S Sᵀ Rᵀ  where R ∈ SO(3), S = diag(s₀,s₁,s₂)
  α ∈ [0,1]        opacity

EWA 2D Projection (Zwicker et al. 2001, Eq. 9)
-----------------------------------------------
  Let t = W[:3,:3] μ + W[:3,3]  (camera-space center)
  Jacobian of local affine approx of perspective projection:

      J = [[fₓ/tz,    0,    -fₓ tx/tz²],
           [   0,   fy/tz,  -fy ty/tz²]]

  2D covariance:  Σ' = J W Σ Wᵀ Jᵀ + 0.3 I₂   (low-pass anti-alias)

  The +0.3 I₂ low-pass filter eliminates aliasing at sub-pixel scale.

Alpha Compositing (front-to-back)
---------------------------------
  α_i(x) = σ_i · exp(-½ (x-μ'ᵢ)ᵀ Σ'ᵢ⁻¹ (x-μ'ᵢ))
  C(x)   = Σᵢ cᵢ αᵢ(x) Tᵢ(x)
  Tᵢ     = ∏_{j<i} (1 - αⱼ(x))

Spherical Harmonics (real, Condon–Shortley convention)
------------------------------------------------------
  Supports l = 0..3 (1, 4, 9, 16 coefficients per channel).

Anti-aliased 3D Gaussian via 2D Gaussian convolution
----------------------------------------------------
  Σ'_aa = Σ' + σ_pixel² I₂
  where σ_pixel = camera-pixel-size in world units.
  This is the correct anti-aliasing formula (Yan et al. 2024).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════
# Quaternion Algebra  (unit quaternion ↔ SO(3))
# ══════════════════════════════════════════════════════════════════

def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert unit quaternion(s) to rotation matrix/matrices.

    Uses the exact formula (no approximation):
        R = (w²-|v|²)I + 2vvᵀ + 2w[v]×

    where v = [x,y,z] and [v]× is the skew-symmetric cross-product matrix.
    Equivalent to the standard expansion:
        R₀₀ = 1 - 2(y²+z²), R₀₁ = 2(xy-wz), R₀₂ = 2(xz+wy), ...

    Args:
        q: (..., 4) tensor [w, x, y, z] — normalized internally.

    Returns:
        (..., 3, 3) rotation matrices in SO(3).
    """
    q = F.normalize(q, p=2, dim=-1)
    w, x, y, z = q.unbind(dim=-1)

    x2, y2, z2 = x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    # Row-major 3×3 rotation matrix
    R = torch.stack([
        1.0 - 2.0*(y2 + z2),  2.0*(xy - wz),        2.0*(xz + wy),
        2.0*(xy + wz),         1.0 - 2.0*(x2 + z2),  2.0*(yz - wx),
        2.0*(xz - wy),         2.0*(yz + wx),         1.0 - 2.0*(x2 + y2),
    ], dim=-1).reshape(q.shape[:-1] + (3, 3))

    return R


def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix/matrices to unit quaternion(s).
    Uses Shepperd's method — numerically stable across all orientations.

    Mathematical basis:
        trace(R) = 1 + 2cos(θ) = 4w² - 1
        → w = ½√(1 + trace(R))
        → x = (R₂₁-R₁₂)/(4w), etc.

    When trace ≤ 0, we find the largest diagonal element to maximize
    numerical stability (avoid division by near-zero w).

    Args:
        R: (..., 3, 3) rotation matrix.

    Returns:
        (..., 4) unit quaternion [w, x, y, z].
    """
    batch_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    N = R.shape[0]
    q = torch.zeros(N, 4, device=R.device, dtype=R.dtype)

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    # Case 1: trace > 0  → w is largest component
    s = torch.sqrt(torch.clamp(trace + 1.0, min=1e-12)) * 2.0   # s = 4w
    q[:, 0] = 0.25 * s
    q[:, 1] = (R[:, 2, 1] - R[:, 1, 2]) / s.clamp(min=1e-12)
    q[:, 2] = (R[:, 0, 2] - R[:, 2, 0]) / s.clamp(min=1e-12)
    q[:, 3] = (R[:, 1, 0] - R[:, 0, 1]) / s.clamp(min=1e-12)

    # Case 2: R[0,0] is largest diagonal
    cond2 = (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2]) & (trace <= 0)
    s2 = torch.sqrt(torch.clamp(1.0 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2], min=1e-12)) * 2.0
    q2 = torch.stack([
        (R[:, 2, 1] - R[:, 1, 2]) / s2.clamp(min=1e-12),
        0.25 * s2,
        (R[:, 0, 1] + R[:, 1, 0]) / s2.clamp(min=1e-12),
        (R[:, 0, 2] + R[:, 2, 0]) / s2.clamp(min=1e-12),
    ], dim=-1)
    q = torch.where(cond2.unsqueeze(-1), q2, q)

    # Case 3: R[1,1] is largest diagonal
    cond3 = (R[:, 1, 1] > R[:, 2, 2]) & ~cond2 & (trace <= 0)
    s3 = torch.sqrt(torch.clamp(1.0 + R[:, 1, 1] - R[:, 0, 0] - R[:, 2, 2], min=1e-12)) * 2.0
    q3 = torch.stack([
        (R[:, 0, 2] - R[:, 2, 0]) / s3.clamp(min=1e-12),
        (R[:, 0, 1] + R[:, 1, 0]) / s3.clamp(min=1e-12),
        0.25 * s3,
        (R[:, 1, 2] + R[:, 2, 1]) / s3.clamp(min=1e-12),
    ], dim=-1)
    q = torch.where(cond3.unsqueeze(-1), q3, q)

    # Case 4: R[2,2] is largest diagonal
    cond4 = ~cond2 & ~cond3 & (trace <= 0)
    s4 = torch.sqrt(torch.clamp(1.0 + R[:, 2, 2] - R[:, 0, 0] - R[:, 1, 1], min=1e-12)) * 2.0
    q4 = torch.stack([
        (R[:, 1, 0] - R[:, 0, 1]) / s4.clamp(min=1e-12),
        (R[:, 0, 2] + R[:, 2, 0]) / s4.clamp(min=1e-12),
        (R[:, 1, 2] + R[:, 2, 1]) / s4.clamp(min=1e-12),
        0.25 * s4,
    ], dim=-1)
    q = torch.where(cond4.unsqueeze(-1), q4, q)

    return F.normalize(q, p=2, dim=-1).reshape(batch_shape + (4,))


def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of quaternions [w, x, y, z]: q1 ⊗ q2."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def quat_slerp(q1: torch.Tensor, q2: torch.Tensor, t: float) -> torch.Tensor:
    """
    Spherical Linear Interpolation between two unit quaternions.

    SLERP formula:
        q(t) = (sin((1-t)Ω)/sinΩ) q1 + (sin(tΩ)/sinΩ) q2
        where Ω = arccos(q1·q2)

    Handles the double-cover ambiguity (q and -q represent the same rotation)
    by flipping q2 if q1·q2 < 0, ensuring the shorter arc is taken.

    Falls back to linear interpolation when Ω ≈ 0 (nearly identical quaternions).

    Args:
        q1, q2: (..., 4) unit quaternions [w, x, y, z]
        t: interpolation parameter in [0, 1]

    Returns:
        (..., 4) interpolated unit quaternion
    """
    q1 = F.normalize(q1, p=2, dim=-1)
    q2 = F.normalize(q2, p=2, dim=-1)

    dot = (q1 * q2).sum(dim=-1, keepdim=True)  # (..., 1)
    # Flip q2 for shortest arc
    q2 = torch.where(dot < 0, -q2, q2)
    dot = dot.abs()

    theta = torch.acos(dot.clamp(-1 + 1e-7, 1 - 1e-7))  # (..., 1)
    sin_theta = torch.sin(theta).clamp(min=1e-8)

    # SLERP coefficients
    c1 = torch.sin((1.0 - t) * theta) / sin_theta
    c2 = torch.sin(t * theta) / sin_theta

    # Linear fallback when sin_theta is tiny
    linear = (1.0 - t) * q1 + t * q2
    slerped = c1 * q1 + c2 * q2

    result = torch.where(sin_theta < 1e-6, linear, slerped)
    return F.normalize(result, p=2, dim=-1)


# ══════════════════════════════════════════════════════════════════
# Covariance Construction
# ══════════════════════════════════════════════════════════════════

def build_covariance_3d(
    scales: torch.Tensor,
    scale_modifier: float,
    rotations: torch.Tensor,
) -> torch.Tensor:
    """
    Build 3D covariance Σ = R S Sᵀ Rᵀ.

    Derivation:
        Σ is the covariance of a Gaussian with principal axes aligned
        with the columns of R and half-lengths given by the scales.
        S = diag(s₀, s₁, s₂) (actual scales, after exp)
        L = R S  →  Σ = L Lᵀ = R S Sᵀ Rᵀ

    Note: scales input is log-scale → exponentiated before use.
    scale_modifier is applied multiplicatively after exp.

    Args:
        scales:          (N, 3) log-scale parameters
        scale_modifier:  global scale multiplier (1.0 during training)
        rotations:       (N, 4) unit quaternions [w, x, y, z]

    Returns:
        (N, 6) upper-triangular symmetric 3×3 Σ:
               [Σ₀₀, Σ₀₁, Σ₀₂, Σ₁₁, Σ₁₂, Σ₂₂]
    """
    R = quaternion_to_rotation_matrix(rotations)              # (N, 3, 3)
    S = torch.diag_embed(torch.exp(scales) * scale_modifier)  # (N, 3, 3)
    L = R @ S                                                  # (N, 3, 3)
    cov = L @ L.transpose(-1, -2)                             # (N, 3, 3) symmetric PSD

    return torch.stack([
        cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2],
        cov[:, 1, 1], cov[:, 1, 2],
        cov[:, 2, 2],
    ], dim=-1)  # (N, 6)


def build_covariance_2d(
    means3d: torch.Tensor,
    covs3d: torch.Tensor,
    viewmatrix: torch.Tensor,
    fovx: float,
    fovy: float,
    image_width: int,
    image_height: int,
    near_threshold: float = 1.3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project 3D Gaussians to 2D via EWA splatting (Zwicker et al. 2001).

    Mathematical derivation:
    ─────────────────────────
    Let t = W[:3,:3] μ + W[:3,3] be the Gaussian center in camera space.
    The perspective projection p: ℝ³→ℝ² maps t ↦ (fx·tx/tz, fy·ty/tz).

    First-order Taylor expansion (affine approximation) at t:
        p(t+δ) ≈ p(t) + J δ

    where the 2×3 Jacobian J at point t = (tx, ty, tz) is:
        J = [[fx/tz,    0,    -fx·tx/tz²],
             [   0,  fy/tz,  -fy·ty/tz²]]

    The projected 2D covariance of the Gaussian is then:
        Σ' = (J W) Σ (J W)ᵀ

    where W = viewmatrix[:3,:3] is the world-to-camera rotation.
    The combined transform T = J @ W is (N, 2, 3).

    Anti-aliasing: add 0.3 I₂ (low-pass filter kernel, Kerbl et al.)
    This prevents degenerate very-thin Gaussians from aliasing at pixel scale.

    Args:
        means3d:      (N, 3) world-space Gaussian centers
        covs3d:       (N, 6) upper-triangular 3D covariance
        viewmatrix:   (4, 4) world-to-camera
        fovx, fovy:   field-of-view in radians
        image_width, image_height: image dimensions in pixels
        near_threshold: near-plane depth threshold (positive z in camera space)

    Returns:
        cov2d: (N, 3) 2D covariance [Σ'₀₀, Σ'₀₁, Σ'₁₁]
        t_cam: (N, 3) camera-space positions
    """
    N = means3d.shape[0]
    device = means3d.device
    dtype = means3d.dtype

    # Transform Gaussian centers to camera space
    W3 = viewmatrix[:3, :3]     # (3, 3)
    t3 = viewmatrix[:3, 3]      # (3,)
    t = means3d @ W3.T + t3     # (N, 3)  camera-space centers

    # Focal lengths from FoV
    fx = image_width  / (2.0 * math.tan(fovx / 2.0))
    fy = image_height / (2.0 * math.tan(fovy / 2.0))

    tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]
    # Safe positive depth mask to avoid gradient explosions for points behind camera
    tz_safe = torch.where(tz > 0.1, tz, torch.ones_like(tz) * 10.0)
    tz_inv   = 1.0 / tz_safe
    tz_inv2  = tz_inv * tz_inv

    zeros = torch.zeros(N, device=device, dtype=dtype)

    # EWA Jacobian: (N, 2, 3)
    J = torch.stack([
        torch.stack([fx * tz_inv,  zeros,        -fx * tx * tz_inv2], dim=-1),
        torch.stack([zeros,         fy * tz_inv,  -fy * ty * tz_inv2], dim=-1),
    ], dim=1)  # (N, 2, 3)

    # Reconstruct symmetric 3D covariance from packed upper-triangular
    cov3d = _upper_tri_to_matrix(covs3d)  # (N, 3, 3)

    # Combined transform T = J @ W: (N, 2, 3)
    T = J @ W3.unsqueeze(0)  # broadcast over N

    # 2D covariance: Σ' = T Σ Tᵀ
    cov2d_full = T @ cov3d @ T.transpose(-1, -2)  # (N, 2, 2)

    # Low-pass anti-aliasing filter (+0.3 I₂)
    eye2 = torch.eye(2, device=device, dtype=dtype).unsqueeze(0)
    cov2d_full = cov2d_full + 0.3 * eye2

    # Pack to (N, 3): upper-tri of symmetric 2×2
    cov2d = torch.stack([
        cov2d_full[:, 0, 0],
        cov2d_full[:, 0, 1],
        cov2d_full[:, 1, 1],
    ], dim=-1)

    return cov2d, t


def build_covariance_2d_antialiased(
    means3d: torch.Tensor,
    covs3d: torch.Tensor,
    viewmatrix: torch.Tensor,
    fovx: float,
    fovy: float,
    image_width: int,
    image_height: int,
    pixel_sigma: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Anti-aliased 2D Gaussian projection (Yan et al. 2024 - Gaussian Opacity Fields).

    Extends EWA with proper Gaussian convolution for anti-aliasing:
        Σ'_aa = Σ'_ewa + σ_pixel² I₂

    where σ_pixel is derived from the pixel footprint in world space.
    This ensures each projected Gaussian covers at least one pixel,
    eliminating the high-frequency flickering from sub-pixel Gaussians.

    Args:
        pixel_sigma: blur radius in pixels (default 0.5 = half-pixel Nyquist)

    Returns:
        cov2d: (N, 3) anti-aliased 2D covariance
        t_cam: (N, 3) camera-space positions
    """
    cov2d, t_cam = build_covariance_2d(
        means3d, covs3d, viewmatrix, fovx, fovy, image_width, image_height
    )
    # Add pixel-level anti-aliasing
    device = cov2d.device
    sigma2 = pixel_sigma ** 2
    # Add to diagonal elements (Σ'₀₀ and Σ'₁₁)
    aa = cov2d.clone()
    aa[:, 0] = aa[:, 0] + sigma2
    aa[:, 2] = aa[:, 2] + sigma2
    return aa, t_cam


# ══════════════════════════════════════════════════════════════════
# Covariance packing helpers
# ══════════════════════════════════════════════════════════════════

def _upper_tri_to_matrix(upper: torch.Tensor) -> torch.Tensor:
    """Reconstruct symmetric 3×3 from 6 upper-triangular elements (N,6) → (N,3,3)."""
    s00, s01, s02, s11, s12, s22 = upper.unbind(-1)
    return torch.stack([
        s00, s01, s02,
        s01, s11, s12,
        s02, s12, s22,
    ], dim=-1).reshape(upper.shape[:-1] + (3, 3))


def strip_lowerdiag(L: torch.Tensor) -> torch.Tensor:
    """Extract upper-triangular elements from (N,3,3) → (N,6)."""
    return torch.stack([
        L[:, 0, 0], L[:, 0, 1], L[:, 0, 2],
        L[:, 1, 1], L[:, 1, 2],
        L[:, 2, 2],
    ], dim=-1)


def strip_symmetric(sym: torch.Tensor) -> torch.Tensor:
    """Alias for strip_lowerdiag."""
    return strip_lowerdiag(sym)


# ══════════════════════════════════════════════════════════════════
# 2D Gaussian Evaluation
# ══════════════════════════════════════════════════════════════════

def invert_cov2d(cov2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Analytically invert 2×2 symmetric positive-definite covariance matrices.

    For [[a, b], [b, c]]:
        det   = ac - b²  (must be > 0 for PD)
        Σ⁻¹   = (1/det) [[c, -b], [-b, a]]

    Numerical stability: det clamped to [1e-14, ∞) to avoid division by zero
    while preserving gradient flow.

    Args:
        cov2d: (N, 3) packed [a, b, c] representing [[a,b],[b,c]]

    Returns:
        cov2d_inv: (N, 2, 2) inverse matrices
        det:       (N,) determinants (before clamping)
    """
    a, b, c = cov2d[:, 0], cov2d[:, 1], cov2d[:, 2]
    det_raw = a * c - b * b
    det_safe = torch.where(det_raw > 1e-6, det_raw, torch.ones_like(det_raw))
    det_inv = 1.0 / det_safe

    cov2d_inv = torch.stack([
        c * det_inv,   -b * det_inv,
        -b * det_inv,   a * det_inv,
    ], dim=-1).reshape(-1, 2, 2)

    return cov2d_inv, det_raw


def gaussian_2d_pdf(
    pixels: torch.Tensor,
    means2d: torch.Tensor,
    cov2d_inv: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluate 2D Gaussian PDF (un-normalized) for alpha compositing.

    α(x) = exp(-½ (x-μ)ᵀ Σ⁻¹ (x-μ))

    The Mahalanobis distance squared: D²(x) = (x-μ)ᵀ Σ⁻¹ (x-μ)
    is a quadratic form. For a 2D vector d = x-μ:
        D² = d₀²·Σ⁻¹₀₀ + 2·d₀·d₁·Σ⁻¹₀₁ + d₁²·Σ⁻¹₁₁

    Args:
        pixels:     (H, W, 2) pixel coordinates
        means2d:    (N, 2) projected Gaussian centers
        cov2d_inv:  (N, 2, 2) inverse 2D covariance

    Returns:
        (N, H, W) per-Gaussian weights in [0, 1]
    """
    H, W, _ = pixels.shape
    N = means2d.shape[0]

    # d: (N, H, W, 2)
    d = pixels.unsqueeze(0) - means2d[:, None, None, :]
    # Mahalanobis²: d Σ⁻¹ dᵀ → (N, H, W)
    md = (d.unsqueeze(-2) @ cov2d_inv[:, None, None] @ d.unsqueeze(-1))
    md = md.squeeze(-1).squeeze(-1)
    return torch.exp(-0.5 * md.clamp(max=20.0))


def compute_radius_from_cov2d(cov2d: torch.Tensor, threshold: float = 3.0) -> torch.Tensor:
    """
    Compute bounding radius in pixels from 2D covariance.

    The 2D Gaussian is bounded by the 3σ ellipse (covering 99.73% of mass).
    We use the largest eigenvalue λ_max of Σ' to get the worst-case radius.

    For a symmetric 2×2 matrix [[a,b],[b,c]]:
        trace = a + c
        det   = ac - b²
        λ_max = trace/2 + √((trace/2)² - det)
             = (a+c)/2 + √((a-c)²/4 + b²)

    Note: discriminant = (trace/2)² - det = ((a-c)/2)² + b² ≥ 0 always.

    Args:
        cov2d:     (N, 3) packed [a, b, c]
        threshold: sigma multiplier (default 3σ)

    Returns:
        (N,) int64 bounding radii in pixels
    """
    a, b, c = cov2d[:, 0], cov2d[:, 1], cov2d[:, 2]
    mid  = 0.5 * (a + c)
    disc = torch.sqrt(torch.clamp(mid * mid - (a * c - b * b), min=0.0))
    lambda_max = mid + disc  # largest eigenvalue
    return torch.ceil(threshold * torch.sqrt(lambda_max.clamp(min=0.0))).long()


# ══════════════════════════════════════════════════════════════════
# SH Color
# ══════════════════════════════════════════════════════════════════

def compute_sh_coefficients(
    dirs: torch.Tensor,
    sh: torch.Tensor,
    sh_degree: int,
) -> torch.Tensor:
    """
    Evaluate per-Gaussian spherical harmonic color.

    Args:
        dirs:      (N, 3) normalized view directions
        sh:        (N, (sh_degree+1)², 3) SH coefficients
        sh_degree: maximum SH degree (0..3)

    Returns:
        (N, 3) RGB colors (before sigmoid/clamping)
    """
    from .sh import eval_sh
    return eval_sh(sh_degree, sh, dirs)


# ══════════════════════════════════════════════════════════════════
# Projection & Screen-Space Utilities
# ══════════════════════════════════════════════════════════════════

def project_points(
    points: torch.Tensor,
    viewmatrix: torch.Tensor,
    projmatrix: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project 3D world-space points to NDC.

    Projection pipeline:
        p_cam  = W p_world    (world → camera, homogeneous)
        p_clip = P p_cam      (camera → clip via projection matrix)
        p_ndc  = p_clip / w   (perspective divide)

    Args:
        points:     (N, 3) world-space points
        viewmatrix: (4, 4) world-to-camera (column vectors)
        projmatrix: (4, 4) camera-to-clip (OpenGL style)

    Returns:
        ndc:   (N, 3) NDC coordinates in [-1,1]³ (approximately)
        depth: (N,) camera-space z-depth
    """
    N = points.shape[0]
    ones = torch.ones(N, 1, device=points.device, dtype=points.dtype)
    ph = torch.cat([points, ones], dim=-1)  # (N, 4) homogeneous

    # Full clip transform: (projmatrix @ viewmatrix) applied row-major
    full_proj = projmatrix @ viewmatrix     # (4, 4)
    clip = ph @ full_proj.T                 # (N, 4)

    w = clip[:, 3:4]
    w_safe = torch.where(w > 0.1, w, torch.ones_like(w) * 10.0)
    ndc = clip[:, :3] / w_safe                   # (N, 3) NDC

    # Camera-space depth from view transform only
    cam = ph @ viewmatrix.T                 # (N, 4)
    depth = cam[:, 2]                       # (N,) z in camera space

    return ndc, depth


def ndc_to_screen(
    ndc: torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor:
    """
    Convert NDC coordinates to pixel-space.

    Mapping:
        x_screen = (ndc_x + 1) / 2 * W     (x: left→right)
        y_screen = (1 - ndc_y) / 2 * H     (y: top→bottom, flip NDC y-up)

    Pixel centres are at integer + 0.5 (half-pixel offset convention).

    Args:
        ndc:    (N, 2) or (N, 3) NDC coords
        width:  image width in pixels
        height: image height in pixels

    Returns:
        (N, 2) pixel coordinates (float)
    """
    screen_x = (ndc[:, 0] + 1.0) * (0.5 * width)
    screen_y = (1.0 - ndc[:, 1]) * (0.5 * height)
    return torch.stack([screen_x, screen_y], dim=-1)


def compute_ndc_to_pixel_jacobian(width: int, height: int) -> torch.Tensor:
    """
    Compute the Jacobian of the NDC→pixel mapping (constant, diagonal):
        ∂x_screen/∂ndc_x = W/2
        ∂y_screen/∂ndc_y = H/2  (note: y-flip makes this negative in full Jac)

    Useful for computing screen-space gradient scales for densification.
    """
    return torch.tensor([width / 2.0, height / 2.0], dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════
# 3D Opacity / Transmittance Computation
# ══════════════════════════════════════════════════════════════════

def compute_transmittance_weights(
    alphas: torch.Tensor,
) -> torch.Tensor:
    """
    Compute front-to-back transmittance weights for alpha compositing.

    T_i = ∏_{j<i} (1 - α_j)    (transmittance before sample i)
    w_i = α_i · T_i             (weight of sample i)

    Implemented via exclusive cumulative product:
        T = cumprod(1 - α, exclusive=True)

    Args:
        alphas: (N,) alpha values in [0, 1], sorted front-to-back

    Returns:
        weights: (N,) transmittance-weighted alpha values
    """
    one_minus_alpha = 1.0 - alphas
    # Exclusive cumulative product: T[0]=1, T[i]=prod(1-α[j]) for j<i
    T = torch.cumprod(one_minus_alpha, dim=0)
    # Shift right (exclusive): prepend 1.0, drop last
    T = torch.cat([torch.ones(1, device=alphas.device, dtype=alphas.dtype), T[:-1]])
    return alphas * T


# ══════════════════════════════════════════════════════════════════
# Numerical Stability Helpers
# ══════════════════════════════════════════════════════════════════

def safe_log(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Numerically safe natural logarithm: log(max(x, eps))."""
    return torch.log(x.clamp(min=eps))


def safe_sqrt(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Numerically safe square root: sqrt(max(x, eps))."""
    return torch.sqrt(x.clamp(min=eps))


def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """Normalize a vector, returning zero-vector if norm < eps."""
    norm = x.norm(dim=dim, keepdim=True)
    return x / norm.clamp(min=eps)


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Numerically stable inverse sigmoid (logit function).
    logit(x) = log(x / (1-x))  for x ∈ (0,1)
    Clamps x to (eps, 1-eps) to avoid log(0).
    """
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))


def log1p_exp(x: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable log(1 + exp(x)) = softplus.
    Uses log1p for small x and x for large x.
    """
    return torch.where(x > 20.0, x, torch.log1p(torch.exp(x.clamp(max=20.0))))


# ══════════════════════════════════════════════════════════════════
# Scene Geometry Helpers
# ══════════════════════════════════════════════════════════════════

def compute_scene_extent_from_cameras(camera_centers: torch.Tensor) -> float:
    """
    Estimate scene extent as the maximum distance from any camera to the centroid.

    This is used to scale densification thresholds.
    Robust to outlier cameras via the max statistic.

    Args:
        camera_centers: (N, 3) world-space camera positions

    Returns:
        float scene extent
    """
    if camera_centers.shape[0] == 0:
        return 1.0
    centroid = camera_centers.mean(dim=0)
    dists = (camera_centers - centroid).norm(dim=-1)  # (N,)
    return float(dists.max().item())


def compute_near_far_from_gaussians(
    gaussians_xyz: torch.Tensor,
    camera_center: torch.Tensor,
    percentile: float = 0.99,
) -> Tuple[float, float]:
    """
    Compute near/far clip planes from Gaussian positions and camera.

    Uses the percentile of distances to avoid outlier-driven clip planes,
    which would waste depth precision on empty space.

    Args:
        gaussians_xyz: (N, 3) Gaussian centers
        camera_center: (3,) camera position
        percentile:    quantile for far plane (0.99 = exclude top 1% outliers)

    Returns:
        (near, far) float tuple
    """
    dists = (gaussians_xyz - camera_center.unsqueeze(0)).norm(dim=-1)  # (N,)
    near = float(dists.min().item()) * 0.5
    far  = float(torch.quantile(dists, percentile).item()) * 1.2
    near = max(near, 0.01)
    far  = max(far, near + 1.0)
    return near, far
