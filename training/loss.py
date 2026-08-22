"""
Loss Functions for 3D Gaussian Splatting Training.

All losses are mathematically documented with their derivations.

Combined Training Loss
──────────────────────
L = (1-λ) · L1(pred, gt)  +  λ · DSSIM(pred, gt)

where DSSIM = (1 - SSIM) / 2  ∈ [0, 0.5].
λ = 0.2 per Kerbl et al. 2023.

SSIM Formula (Wang et al. 2004)
───────────────────────────────
For local patches around each pixel:
  SSIM(x,y) = L(x,y) · C(x,y) · S(x,y)

where:
  L(x,y) = (2μ_xμ_y + C1) / (μ_x² + μ_y² + C1)    (luminance)
  C(x,y) = (2σ_xσ_y + C2) / (σ_x² + σ_y² + C2)    (contrast)
  S(x,y) = (σ_xy + C3) / (σ_xσ_y + C3)             (structure)

The standard implementation combines L and C·S:
  SSIM(x,y) = (2μ_xμ_y + C1)(σ_xy·κ + C2) /
              ((μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2))

with C1=(0.01·L)², C2=(0.03·L)², L=1 for images in [0,1].

Local statistics are computed via separable Gaussian convolution
with window_size=11, σ=1.5 (Wang et al. standard).

Scale-Invariant Depth Loss (Eigen et al. 2014)
──────────────────────────────────────────────
  L_SI = (1/n)Σ δᵢ² - (λ/n²)(Σ δᵢ)²
  δᵢ = log(dˆᵢ) - log(dᵢ)
  λ = 0.5 gives full scale-invariance.

Opacity Entropy Regularization
──────────────────────────────
Encourages binary opacity (either fully transparent or opaque):
  L_ent = -Σ(α log α + (1-α) log(1-α)) / N

This pushes opacities toward {0, 1}, which improves
scene geometry reconstruction quality.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════
# Basic Losses
# ══════════════════════════════════════════════════════════════════

def l1_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Error (L1 norm): E[|pred - gt|]."""
    return (pred - gt).abs().mean()


def l2_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Mean Squared Error (L2 norm squared): E[(pred - gt)²]."""
    return ((pred - gt) ** 2).mean()


def huber_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    delta: float = 0.1,
) -> torch.Tensor:
    """
    Huber (smooth-L1) loss: interpolates between L1 and L2.

    L_δ(x) = { ½ x²         if |x| ≤ δ
              { δ(|x| - ½δ)  otherwise

    More robust to outliers than L2, smoother than L1.
    """
    diff = (pred - gt).abs()
    return torch.where(
        diff <= delta,
        0.5 * diff ** 2,
        delta * (diff - 0.5 * delta)
    ).mean()


# ══════════════════════════════════════════════════════════════════
# SSIM
# ══════════════════════════════════════════════════════════════════

def _gaussian_window(
    window_size: int,
    sigma: float,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Build 2D Gaussian kernel for SSIM computation.

    Construction:
        1D:  g(k) = exp(-(k - c)² / (2σ²))   for k = 0..W-1, c = W//2
        2D:  G = g(x) ⊗ g(y)^T  (separable outer product)
             then normalized to sum to 1.
    The kernel is replicated for grouped channel-wise convolution.

    Args:
        window_size: size of the Gaussian window (default 11)
        sigma:       Gaussian σ (default 1.5)
        channels:    number of image channels (for grouped conv)
        device, dtype: output tensor properties

    Returns:
        (C, 1, W, W) kernel
    """
    center = window_size // 2
    coords = torch.arange(window_size, dtype=dtype, device=device) - center
    g1d = torch.exp(-coords ** 2 / (2.0 * sigma ** 2))
    g1d = g1d / g1d.sum()         # normalize to sum 1
    g2d = g1d.unsqueeze(-1) @ g1d.unsqueeze(0)  # (W, W) outer product
    g2d = g2d / g2d.sum()         # normalize 2D (already normalized, but safe)
    return g2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, window_size, window_size)


def ssim_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """
    Structural Similarity Index (SSIM) loss = 1 - SSIM(pred, gt).

    Full SSIM formula (Wang et al. 2004, IEEE TIP):
        SSIM = (2μ_xμ_y + C1)(σ_xy·κ + C2) /
               ((μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2))

    where:
        C1 = (K1·L)², C2 = (K2·L)² (stability constants)
        K1=0.01, K2=0.03, L=1 for [0,1] images
        μ_x = E[x], σ_x² = E[x²] - μ_x² (local Gaussian-weighted stats)
        σ_xy = E[xy] - μ_xμ_y

    Note: σ_x² = Var(x) can be slightly negative due to float precision.
    Clamping the denominator to [eps, ∞) prevents NaN gradients.

    Args:
        pred: (C, H, W) or (B, C, H, W) predicted image in [0, 1]
        gt:   same shape as pred
        window_size: Gaussian window size (default 11 per Wang et al.)
        sigma:       Gaussian σ (default 1.5 per Wang et al.)

    Returns:
        scalar SSIM loss in [0, 1] (lower = more similar)
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        gt   = gt.unsqueeze(0)

    B, C, H, W = pred.shape
    device, dtype = pred.device, pred.dtype

    window = _gaussian_window(window_size, sigma, C, device, dtype)
    pad = window_size // 2

    def _conv(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, window, padding=pad, groups=C)

    mu_x   = _conv(pred)
    mu_y   = _conv(gt)
    mu_x2  = mu_x * mu_x
    mu_y2  = mu_y * mu_y
    mu_xy  = mu_x * mu_y

    # Variances and covariance
    sig_x2  = _conv(pred * pred) - mu_x2
    sig_y2  = _conv(gt   * gt)   - mu_y2
    sig_xy  = _conv(pred * gt)   - mu_xy

    # SSIM constants
    K1, K2, L = 0.01, 0.03, 1.0
    C1 = (K1 * L) ** 2  # 1e-4
    C2 = (K2 * L) ** 2  # 9e-4

    numerator   = (2.0 * mu_xy + C1) * (2.0 * sig_xy + C2)
    denominator = (mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2)

    ssim_map = numerator / denominator.clamp(min=1e-10)
    return 1.0 - ssim_map.mean()


def dssim(
    pred: torch.Tensor,
    gt: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """DSSIM = (1 - SSIM) / 2  ∈ [0, 0.5]. As used in 3DGS paper."""
    return ssim_loss(pred, gt, **kwargs) * 0.5


def combined_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    lambda_dssim: float = 0.2,
) -> torch.Tensor:
    """
    Standard official 3DGS combined loss (Kerbl et al. 2023):

        L = (1 - λ) · L1(pred, gt)  +  λ · (1 - SSIM(pred, gt))

    Args:
        pred:         (3, H, W) or (B, 3, H, W) rendered image in [0, 1]
        gt:           same shape, ground-truth
        lambda_dssim: DSSIM weight (default 0.2)

    Returns:
        scalar combined loss
    """
    l1 = l1_loss(pred, gt)
    ssim_val = ssim_loss(pred, gt)
    return (1.0 - lambda_dssim) * l1 + lambda_dssim * ssim_val


# ══════════════════════════════════════════════════════════════════
# Depth Losses
# ══════════════════════════════════════════════════════════════════

def perceptual_depth_loss(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    lambda_si: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Scale-invariant logarithmic depth loss (Eigen et al. NeurIPS 2014).

    L_SI = (1/n)Σ δᵢ² - (λ/n²)(Σ δᵢ)²
    where δᵢ = log(dˆᵢ) - log(dᵢ)

    Mathematical properties:
        - Scale-invariant: L_SI(s·dˆ, s·d) = L_SI(dˆ, d) for any s > 0
        - Reduces to MSE of log-depths when λ = 0
        - λ = 0.5 gives the full scale-invariant formulation
        - Equivalent to minimizing variance of per-pixel log-depth errors

    Args:
        pred_depth: (H, W) or (B, H, W) predicted depth (must be > 0)
        gt_depth:   same shape, ground-truth depth (must be > 0)
        lambda_si:  scale-invariance weight (0.5 = standard)
        eps:        minimum valid depth value

    Returns:
        scalar depth loss
    """
    valid = (gt_depth > eps) & (pred_depth > eps)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred_depth.device, dtype=pred_depth.dtype)

    log_diff = torch.log(pred_depth[valid] + eps) - torch.log(gt_depth[valid] + eps)
    n = log_diff.numel()
    # Scale-invariant term
    si_correction = lambda_si * log_diff.sum() ** 2 / max(n ** 2, 1)
    return (log_diff ** 2).mean() - si_correction


def affine_depth_loss(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Affine-invariant depth loss (MiDaS-style, Ranftl et al. 2020).

    Aligns pred to gt via least-squares affine transform before computing loss.
    Useful when the depth scale/shift is unknown (e.g., monocular priors).

    Alignment: find (s, t) minimizing sum((s*pred + t - gt)^2)
    Then compute: L = MSE(s*pred + t, gt)

    Args:
        pred_depth: (H, W) predicted depth
        gt_depth:   (H, W) ground truth depth
        eps:        valid depth threshold

    Returns:
        scalar loss
    """
    valid = (gt_depth > eps) & (pred_depth > eps)
    if valid.sum() < 10:
        return torch.tensor(0.0, device=pred_depth.device, dtype=pred_depth.dtype)

    p = pred_depth[valid].float()
    g = gt_depth[valid].float()

    # Least-squares: [p, 1] @ [s, t]^T = g
    # Normal equations: (P^T P) [s,t]^T = P^T g
    n = float(p.numel())
    sum_p  = p.sum()
    sum_g  = g.sum()
    sum_pp = (p * p).sum()
    sum_pg = (p * g).sum()

    det = n * sum_pp - sum_p ** 2
    if det.abs() < 1e-10:
        return torch.tensor(0.0, device=pred_depth.device, dtype=pred_depth.dtype)

    s = (n * sum_pg - sum_p * sum_g) / det
    t = (sum_pp * sum_g - sum_p * sum_pg) / det

    aligned = s * p + t
    return F.mse_loss(aligned, g)


# ══════════════════════════════════════════════════════════════════
# Regularization Losses
# ══════════════════════════════════════════════════════════════════

def opacity_entropy_loss(opacities: torch.Tensor) -> torch.Tensor:
    """
    Binary entropy regularization on opacities.

    H(α) = -α log(α) - (1-α) log(1-α)  (binary entropy)

    Minimizing H encourages opacities to be either 0 or 1.
    This improves geometry reconstruction by reducing "foggy" Gaussians
    that partially cover multiple surfaces.

    Args:
        opacities: (N, 1) or (N,) opacity values in (0, 1)

    Returns:
        mean binary entropy (scalar)
    """
    a = opacities.squeeze().clamp(1e-6, 1.0 - 1e-6)
    entropy = -a * torch.log(a) - (1.0 - a) * torch.log(1.0 - a)
    return entropy.mean()


def scale_regularization_loss(
    scales: torch.Tensor,
    max_log_scale: float = 2.0,
) -> torch.Tensor:
    """
    Penalize excessively large Gaussian scales.

    L_scale = mean(relu(max(log_s_i) - max_log_scale)^2)

    This prevents unbounded scale growth which creates large, flat
    Gaussian "discs" that cover large regions with incorrect geometry.

    Args:
        scales:        (N, 3) log-scale parameters (raw, before exp)
        max_log_scale: maximum allowed log-scale (default 2.0 ≈ exp(2)=7.4 units)

    Returns:
        scalar regularization loss
    """
    max_scale = scales.max(dim=-1).values  # (N,) per-Gaussian max scale
    return F.relu(max_scale - max_log_scale).pow(2).mean()


def isotropy_loss(scales: torch.Tensor) -> torch.Tensor:
    """
    Penalize highly anisotropic Gaussians.

    L_iso = mean((max(s) - min(s))^2 / max(s)^2)

    Anisotropic Gaussians ("pancakes" or "needles") can represent
    textures/edges but may also cause rendering artifacts.
    This is a soft regularizer, not a hard constraint.

    Args:
        scales: (N, 3) log-scale parameters

    Returns:
        scalar isotropy loss
    """
    s = torch.exp(scales)  # (N, 3) actual scales
    s_max = s.max(dim=-1).values
    s_min = s.min(dim=-1).values
    ratio = ((s_max - s_min) / s_max.clamp(min=1e-8)) ** 2
    return ratio.mean()


# ══════════════════════════════════════════════════════════════════
# Metrics (not differentiable — for evaluation only)
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def psnr(pred: torch.Tensor, gt: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Peak Signal-to-Noise Ratio.

    PSNR = 10 · log₁₀(MAX² / MSE)

    For images in [0,1]: MAX=1, so PSNR = -10 · log₁₀(MSE).
    Higher PSNR = better reconstruction (30+ dB is good, 40+ is excellent).

    Args:
        pred, gt: tensors in [0, max_val]
        max_val:  maximum signal value (1.0 for normalized images)

    Returns:
        PSNR in dB (float)
    """
    mse = ((pred - gt) ** 2).mean().item()
    if mse < 1e-12:
        return float('inf')
    return 10.0 * math.log10(max_val ** 2 / mse)


@torch.no_grad()
def ssim_metric(
    pred: torch.Tensor,
    gt: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """
    SSIM metric value in [0, 1] (higher = more similar).
    For evaluation only (not for training).
    """
    return float(1.0 - ssim_loss(pred, gt, window_size, sigma).item())


@torch.no_grad()
def lpips_metric(pred: torch.Tensor, gt: torch.Tensor) -> Optional[float]:
    """
    Learned Perceptual Image Patch Similarity (LPIPS).
    Requires the `lpips` package. Returns None if not installed.
    Lower LPIPS = more perceptually similar.
    """
    try:
        import lpips
        loss_fn = lpips.LPIPS(net='alex').to(pred.device)
        # LPIPS expects (B, C, H, W) in [-1, 1]
        p = (pred.unsqueeze(0) * 2 - 1).clamp(-1, 1)
        g = (gt.unsqueeze(0)   * 2 - 1).clamp(-1, 1)
        return float(loss_fn(p, g).mean().item())
    except ImportError:
        return None
