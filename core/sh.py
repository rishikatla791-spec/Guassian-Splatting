"""
Spherical Harmonics for 3D Gaussian Splatting.

Mathematical Basis
──────────────────
Real spherical harmonics Yₗᵐ(θ,φ) form an orthonormal basis on S²:

  ∫_{S²} Yₗᵐ(d) Yₗ'ᵐ'(d) dΩ = δₗₗ' δₘₘ'

A view-dependent color signal C: S²→ℝ³ is expanded as:

  C(d) = Σₗ₌₀^L Σₘ₌₋ₗ^ₗ  cₗᵐ · Yₗᵐ(d)

For l_max=3 we use (l_max+1)² = 16 coefficients per color channel.

Condon–Shortley Phase Convention (used here)
─────────────────────────────────────────────
Real SH basis evaluated at direction d = (x, y, z) ∈ S²:

Band 0 (l=0):
  Y₀⁰  =  C₀  = 0.28209479177387814    (constant)

Band 1 (l=1):
  Y₁⁻¹ = C₁ · y   = 0.4886025119029199 · y
  Y₁⁰  = C₁ · z   = 0.4886025119029199 · z
  Y₁¹  = C₁ · x   = 0.4886025119029199 · x

Band 2 (l=2):
  Y₂⁻² = C₂₀ · xy          = 1.0925484305920792 · xy
  Y₂⁻¹ = C₂₁ · yz          = -1.0925484305920792 · yz   (note sign!)
  Y₂⁰  = C₂₂ · (2z²-x²-y²) = 0.31539156525252005 · (2z²-x²-y²)
  Y₂¹  = C₂₃ · xz          = -1.0925484305920792 · xz   (note sign!)
  Y₂²  = C₂₄ · (x²-y²)     = 0.5462742152960396 · (x²-y²)

Band 3 (l=3):
  Y₃⁻³ = C₃₀ · y(3x²-y²)
  Y₃⁻² = C₃₁ · xyz
  Y₃⁻¹ = C₃₂ · y(4z²-x²-y²)
  Y₃⁰  = C₃₃ · z(2z²-3x²-3y²)
  Y₃¹  = C₃₄ · x(4z²-x²-y²)
  Y₃²  = C₃₅ · z(x²-y²)
  Y₃³  = C₃₆ · x(x²-3y²)

Coefficients are exactly as in:
  Sloan 2008 "Stupid Spherical Harmonics Tricks"
  Ramamoorthi & Hanrahan 2001

Note on numerical precision:
  All SH basis constants are stored as IEEE 754 double-precision
  literals derived from the exact formulae:
    C₀ = 1/(2√π)
    C₁ = √(3/(4π))
    etc.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════
# SH Coefficient Count
# ══════════════════════════════════════════════════════════════════

def num_sh_coefficients(degree: int) -> int:
    """Total number of real SH coefficients for degrees 0..degree: (degree+1)²."""
    return (degree + 1) ** 2


# ══════════════════════════════════════════════════════════════════
# RGB ↔ SH DC Conversion
# ══════════════════════════════════════════════════════════════════

# Y₀⁰ = 1/(2√π)
C0: float = 0.28209479177387814

def RGB2SH(rgb: torch.Tensor) -> torch.Tensor:
    """
    Convert linear RGB to SH DC coefficient (band 0).

    The DC term of the SH expansion is:
        f₀ = ∫ C(d) Y₀⁰ dΩ = C₀ · C_mean

    where C_mean is the mean color. Rearranging:
        SH_dc = (RGB - 0.5) / C₀

    The -0.5 centers the color around zero (SH naturally represents
    functions on the sphere; 0.5 gray maps to zero DC coefficient).
    """
    return (rgb - 0.5) / C0


def SH2RGB(sh: torch.Tensor) -> torch.Tensor:
    """Convert SH DC coefficient back to linear RGB."""
    return sh * C0 + 0.5


# ══════════════════════════════════════════════════════════════════
# SH Basis Constants (exact double-precision)
# ══════════════════════════════════════════════════════════════════

# Band 1: C₁ = √(3/(4π))
C1: float = 0.4886025119029199

# Band 2: derived from associated Legendre polynomials
C2 = [
    1.0925484305920792,    # √(15/(4π)) for Y₂⁻²
   -1.0925484305920792,   # −√(15/(4π)) for Y₂⁻¹ (Condon-Shortley)
    0.31539156525252005,   # ½√(5/π) for Y₂⁰
   -1.0925484305920792,   # −√(15/(4π)) for Y₂¹
    0.5462742152960396,    # ¼√(15/π) for Y₂²
]

# Band 3: derived from associated Legendre polynomials
C3 = [
   -0.5900435899266435,   # −¼√(35/(2π)) for Y₃⁻³
    2.890611442640554,    # √(105/π) for Y₃⁻²
   -0.4570457994644658,   # −¼√(21/(2π)) for Y₃⁻¹
    0.3731763325901154,   # ¼√(7/π) for Y₃⁰
   -0.4570457994644658,   # −¼√(21/(2π)) for Y₃¹
    1.4453057213188614,   # ¼√(105/π) for Y₃²
   -0.5900435899266435,   # −¼√(35/(2π)) for Y₃³
]


# ══════════════════════════════════════════════════════════════════
# Core SH Evaluation
# ══════════════════════════════════════════════════════════════════

def eval_sh(
    degree: int,
    sh: torch.Tensor,
    dirs: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluate real spherical harmonics at view directions.

    Mathematical expansion:
        C(d) = C₀·c₀ + C₁(-y·c₁ + z·c₂ - x·c₃) + (band2 terms) + ...

    Note: the SH coefficients cₗᵐ are stored in row-major order:
        c[0]        = c₀⁰  (band 0)
        c[1..3]     = c₁⁻¹, c₁⁰, c₁¹  (band 1)
        c[4..8]     = c₂⁻², c₂⁻¹, c₂⁰, c₂¹, c₂²  (band 2)
        c[9..15]    = c₃⁻³, ..., c₃³  (band 3)

    The color output is in an un-clamped linear space. Caller should
    add 0.5 and clamp to [0,1] for display:
        RGB = clamp(eval_sh(...) + 0.5, 0, 1)

    Args:
        degree: active SH degree (0..3)
        sh:     (N, K, 3) SH coefficients where K = (degree+1)²
        dirs:   (N, 3) normalized view directions [x, y, z]

    Returns:
        (N, 3) raw RGB color (needs +0.5 and clamping)
    """
    assert 0 <= degree <= 3, f"SH degree must be in [0,3], got {degree}"
    assert sh.shape[1] >= num_sh_coefficients(degree), (
        f"SH tensor has {sh.shape[1]} coefficients but degree {degree} needs {num_sh_coefficients(degree)}"
    )

    # Decompose direction components — ensure normalized
    dirs = F.normalize(dirs, p=2, dim=-1)  # (N, 3) re-normalize for safety
    x = dirs[:, 0:1]  # (N, 1)
    y = dirs[:, 1:2]
    z = dirs[:, 2:3]

    # ── Band 0 (l=0): constant ──────────────────────────────────────
    # C(d) = C₀ · c₀     for all d
    result = C0 * sh[:, 0]   # (N, 3)

    if degree < 1:
        return result

    # ── Band 1 (l=1): linear in d ──────────────────────────────────
    # C₁(-y c₁ + z c₂ - x c₃)
    # SH ordering: Y₁⁻¹ = C₁·y, Y₁⁰ = C₁·z, Y₁¹ = -C₁·x
    result = (
        result
        - C1 * y * sh[:, 1]   # Y₁⁻¹ term
        + C1 * z * sh[:, 2]   # Y₁⁰  term
        - C1 * x * sh[:, 3]   # Y₁¹  term (negative Condon-Shortley)
    )

    if degree < 2:
        return result

    # ── Band 2 (l=2): quadratic in d ───────────────────────────────
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z

    result = (
        result
        + C2[0] * xy          * sh[:, 4]   # Y₂⁻²: xy
        + C2[1] * yz          * sh[:, 5]   # Y₂⁻¹: yz
        + C2[2] * (2*zz-xx-yy) * sh[:, 6]  # Y₂⁰:  2z²-x²-y²
        + C2[3] * xz          * sh[:, 7]   # Y₂¹:  xz
        + C2[4] * (xx - yy)   * sh[:, 8]   # Y₂²:  x²-y²
    )

    if degree < 3:
        return result

    # ── Band 3 (l=3): cubic in d ────────────────────────────────────
    result = (
        result
        + C3[0] * y * (3*xx - yy)        * sh[:, 9]    # Y₃⁻³
        + C3[1] * xy * z                  * sh[:, 10]   # Y₃⁻²
        + C3[2] * y * (4*zz - xx - yy)   * sh[:, 11]   # Y₃⁻¹
        + C3[3] * z * (2*zz - 3*xx - 3*yy) * sh[:, 12] # Y₃⁰
        + C3[4] * x * (4*zz - xx - yy)   * sh[:, 13]   # Y₃¹
        + C3[5] * z * (xx - yy)           * sh[:, 14]   # Y₃²
        + C3[6] * x * (xx - 3*yy)        * sh[:, 15]   # Y₃³
    )

    return result


def eval_sh_grad(
    degree: int,
    sh: torch.Tensor,
    dirs: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluate SH and return the gradient of the color w.r.t. view direction.

    Useful for regularization: penalizing large directional color variation
    encourages the model to use geometry rather than view-dependent color
    to explain scene appearance (anti-floater regularization).

    Returns:
        color: (N, 3)
        grad_dirs: (N, 3, 3)  Jacobian ∂color/∂dir
    """
    dirs_leaf = dirs.detach().requires_grad_(True)
    color = eval_sh(degree, sh, dirs_leaf)
    # Compute Jacobian
    grad_list = []
    for ch in range(3):
        g = torch.autograd.grad(
            color[:, ch].sum(), dirs_leaf,
            retain_graph=(ch < 2), create_graph=False
        )[0]  # (N, 3)
        grad_list.append(g)
    grad_dirs = torch.stack(grad_list, dim=1)  # (N, 3, 3)
    return color.detach(), grad_dirs


def sh_smoothness_loss(sh: torch.Tensor, degree: int = 3) -> torch.Tensor:
    """
    Regularization loss penalizing high-frequency SH content.

    Higher SH bands carry more high-frequency view-dependence.
    We penalize the energy in bands l≥1 relative to band 0:
        L_smooth = Σ_{l=1}^L (l+1)² ||c_l||²

    The (l+1)² weighting penalizes high bands more strongly.

    Args:
        sh:     (N, K, 3) SH coefficients
        degree: active SH degree (0..3)

    Returns:
        scalar regularization loss
    """
    total = torch.tensor(0.0, device=sh.device, dtype=sh.dtype)
    idx = 1  # start after band 0
    for l in range(1, degree + 1):
        n_coeffs = 2 * l + 1  # number of coefficients at degree l
        band_energy = (sh[:, idx:idx+n_coeffs, :] ** 2).sum()
        total = total + (l + 1) ** 2 * band_energy
        idx += n_coeffs
    return total / sh.shape[0]  # normalize by N
