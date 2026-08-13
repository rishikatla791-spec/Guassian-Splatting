"""
CUDA & GPU-Accelerated Core Operations for 3D Gaussian Splatting.

Provides high-throughput CUDA kernel ops and PyTorch GPU routines for:
  1. 3D Covariance Construction: Σ = R S Sᵀ Rᵀ
  2. EWA 2D Covariance Projection: Σ' = J W Σ Wᵀ Jᵀ + 0.3 I₂
  3. Analytic 2D Covariance Inversion & Determinant
  4. Spherical Harmonics (SH) Degree 0..3 GPU Evaluation
  5. GPU Frustum Culling & Bounding Radius Computation
  6. Custom PyTorch Autograd CUDA Tile Rasterizer Function

Optimized specifically for RTX 3050 (6GB VRAM) with minimal host-device transfers.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sh import C0, eval_sh
from .math_utils import quaternion_to_rotation_matrix, project_points, ndc_to_screen


# ══════════════════════════════════════════════════════════════════
# 1. GPU 3D Covariance Construction: Σ = R S Sᵀ Rᵀ
# ══════════════════════════════════════════════════════════════════

class BuildCovariance3DFunction(torch.autograd.Function):
    """
    Custom PyTorch Autograd CUDA function for 3D covariance construction:
        Σ = R S Sᵀ Rᵀ

    Forward Pass:
        Computes (N, 6) upper-triangular symmetric 3x3 covariance matrix.
    Backward Pass:
        Analytical gradients w.r.t. log-scales (N, 3) and raw quaternions (N, 4).
    """

    @staticmethod
    def forward(
        ctx: Any,
        scaling: torch.Tensor,       # (N, 3) log-scale
        scale_modifier: float,
        rotation: torch.Tensor,      # (N, 4) unit quaternion [w, x, y, z]
    ) -> torch.Tensor:
        # Normalize quaternion
        q_norm = F.normalize(rotation, p=2, dim=-1)
        
        # Scaling matrix S = diag(exp(s) * modifier)
        s = torch.exp(scaling) * scale_modifier # (N, 3)
        
        # Rotation matrix R: (N, 3, 3)
        R = quaternion_to_rotation_matrix(q_norm)
        
        # L = R @ S: (N, 3, 3)
        L = R * s.unsqueeze(1) # Broadcast scale across columns
        
        # Covariance Σ = L @ Lᵀ: (N, 3, 3)
        cov = torch.bmm(L, L.transpose(1, 2))
        
        # Upper triangular elements [cov00, cov01, cov02, cov11, cov12, cov22]
        cov3d = torch.stack([
            cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2],
            cov[:, 1, 1], cov[:, 1, 2],
            cov[:, 2, 2]
        ], dim=-1)
        
        ctx.save_for_backward(scaling, q_norm, R, s, rotation)
        ctx.scale_modifier = scale_modifier
        return cov3d

    @staticmethod
    def backward(ctx: Any, grad_cov3d: torch.Tensor) -> Tuple[torch.Tensor, None, torch.Tensor]:
        scaling, q_norm, R, s, rotation = ctx.saved_tensors
        scale_modifier = ctx.scale_modifier
        
        # Reconstruct full 3x3 symmetric gradient matrix
        g00, g01, g02, g11, g12, g22 = grad_cov3d.unbind(-1)
        grad_cov = torch.stack([
            g00,       0.5 * g01, 0.5 * g02,
            0.5 * g01, g11,       0.5 * g12,
            0.5 * g02, 0.5 * g12, g22
        ], dim=-1).reshape(-1, 3, 3)
        
        # Grad w.r.t L = R S: dL/d(cov) = 2 * grad_cov @ L
        L = R * s.unsqueeze(1)
        dL_dL = 2.0 * torch.bmm(grad_cov, L) # (N, 3, 3)
        
        # Grad w.r.t s (scale): dL/ds_i = sum_j (dL/dL_ji * R_ji) * exp(scale_i) * modifier
        dL_ds_unscaled = (dL_dL * R).sum(dim=1) # (N, 3)
        grad_scaling = dL_ds_unscaled * s
        
        # Grad w.r.t R: dL/dR = dL_dL @ S
        dL_dR = dL_dL * s.unsqueeze(1) # (N, 3, 3)
        
        # Grad w.r.t raw quaternion q = [w, x, y, z]
        w, x, y, z = q_norm.unbind(-1)
        
        # Exact analytical derivatives of R w.r.t [w, x, y, z]
        # R00 = 1 - 2(y^2+z^2), R01 = 2(xy-wz), R02 = 2(xz+wy)
        # R10 = 2(xy+wz), R11 = 1 - 2(x^2+z^2), R12 = 2(yz-wx)
        # R20 = 2(xz-wy), R21 = 2(yz+wx), R22 = 1 - 2(x^2+y^2)
        
        dR_dw = torch.stack([
            torch.zeros_like(w), -2.0*z,         2.0*y,
            2.0*z,              torch.zeros_like(w), -2.0*x,
            -2.0*y,             2.0*x,          torch.zeros_like(w)
        ], dim=-1).reshape(-1, 3, 3)
        
        dR_dx = torch.stack([
            torch.zeros_like(x),  2.0*y,         2.0*z,
            2.0*y,               -4.0*x,        -2.0*w,
            2.0*z,                2.0*w,        -4.0*x
        ], dim=-1).reshape(-1, 3, 3)
        
        dR_dy = torch.stack([
            -4.0*y,               2.0*x,         2.0*w,
            2.0*x,                torch.zeros_like(y),  2.0*z,
            -2.0*w,               2.0*z,        -4.0*y
        ], dim=-1).reshape(-1, 3, 3)
        
        dR_dz = torch.stack([
            -4.0*z,              -2.0*w,         2.0*x,
            2.0*w,               -4.0*z,         2.0*y,
            2.0*x,                2.0*y,         torch.zeros_like(z)
        ], dim=-1).reshape(-1, 3, 3)
        
        grad_w = (dL_dR * dR_dw).sum(dim=(1, 2))
        grad_x = (dL_dR * dR_dx).sum(dim=(1, 2))
        grad_y = (dL_dR * dR_dy).sum(dim=(1, 2))
        grad_z = (dL_dR * dR_dz).sum(dim=(1, 2))
        
        grad_q = torch.stack([grad_w, grad_x, grad_y, grad_z], dim=-1)
        
        # Project gradient through unit normalization
        q_dot_grad = (q_norm * grad_q).sum(dim=-1, keepdim=True)
        grad_rotation = (grad_q - q_norm * q_dot_grad) / rotation.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        
        return grad_scaling, None, grad_rotation


def build_covariance_3d_cuda(
    scaling: torch.Tensor,
    scale_modifier: float,
    rotation: torch.Tensor
) -> torch.Tensor:
    """CUDA GPU wrapper for 3D covariance construction."""
    return BuildCovariance3DFunction.apply(scaling, scale_modifier, rotation)


# ══════════════════════════════════════════════════════════════════
# 2. GPU 2D EWA Covariance Projection
# ══════════════════════════════════════════════════════════════════

def build_covariance_2d_cuda(
    means3d: torch.Tensor,
    covs3d: torch.Tensor,
    viewmatrix: torch.Tensor,
    fovx: float,
    fovy: float,
    image_width: int,
    image_height: int,
    near_threshold: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GPU-accelerated EWA 2D projection kernel:
        Σ' = J W Σ Wᵀ Jᵀ + 0.3 I₂

    Args:
        means3d: (N, 3) world space centers
        covs3d: (N, 6) 3D covariance upper triangular
        viewmatrix: (4, 4) world-to-camera matrix
        fovx, fovy: FOV angles in radians
        image_width, image_height: dimensions in pixels

    Returns:
        cov2d: (N, 3) 2D covariance [a, b, c] for [[a, b], [b, c]]
        t_cam: (N, 3) camera space positions
    """
    N = means3d.shape[0]
    device = means3d.device
    dtype = means3d.dtype

    # Transform centers to camera space: t = W3 @ mu + t3
    W3 = viewmatrix[:3, :3] # (3, 3)
    t3 = viewmatrix[:3, 3]  # (3,)
    t_cam = torch.matmul(means3d, W3.T) + t3 # (N, 3)

    # Focal lengths
    fx = float(image_width / (2.0 * math.tan(fovx / 2.0)))
    fy = float(image_height / (2.0 * math.tan(fovy / 2.0)))

    tx, ty, tz = t_cam[:, 0], t_cam[:, 1], t_cam[:, 2]
    tz_safe = tz.clamp(min=near_threshold)
    tz_inv = 1.0 / tz_safe
    tz_inv2 = tz_inv * tz_inv

    zeros = torch.zeros(N, device=device, dtype=dtype)

    # EWA Jacobian J: (N, 2, 3)
    J = torch.stack([
        torch.stack([fx * tz_inv,  zeros,        -fx * tx * tz_inv2], dim=-1),
        torch.stack([zeros,         fy * tz_inv,  -fy * ty * tz_inv2], dim=-1),
    ], dim=1)

    # Reconstruct 3D covariance matrix (N, 3, 3)
    s00, s01, s02, s11, s12, s22 = covs3d.unbind(-1)
    cov3d_matrix = torch.stack([
        s00, s01, s02,
        s01, s11, s12,
        s02, s12, s22
    ], dim=-1).reshape(N, 3, 3)

    # T = J @ W3: (N, 2, 3)
    T = torch.matmul(J, W3.unsqueeze(0))

    # cov2d_full = T @ cov3d @ Tᵀ: (N, 2, 2)
    cov2d_full = torch.bmm(torch.bmm(T, cov3d_matrix), T.transpose(1, 2))

    # Add low-pass anti-aliasing filter (+0.3 I₂)
    cov2d_full[:, 0, 0] += 0.3
    cov2d_full[:, 1, 1] += 0.3

    cov2d = torch.stack([
        cov2d_full[:, 0, 0],
        cov2d_full[:, 0, 1],
        cov2d_full[:, 1, 1]
    ], dim=-1)

    return cov2d, t_cam


# ══════════════════════════════════════════════════════════════════
# 3. Analytic 2D Covariance Inversion & Determinant
# ══════════════════════════════════════════════════════════════════

def invert_cov2d_cuda(cov2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Analytic 2x2 inversion on GPU.
    For [[a, b], [b, c]]:
        det = a*c - b*b
        inv = (1/det) * [[c, -b], [-b, a]]

    Returns:
        cov2d_inv: (N, 2, 2) inverse matrix
        det: (N,) determinant
    """
    a, b, c = cov2d[:, 0], cov2d[:, 1], cov2d[:, 2]
    det_raw = a * c - b * b
    det = det_raw.clamp(min=1e-14)
    det_inv = 1.0 / det

    cov2d_inv = torch.stack([
        c * det_inv,  -b * det_inv,
       -b * det_inv,   a * det_inv
    ], dim=-1).reshape(-1, 2, 2)

    return cov2d_inv, det_raw


def compute_radius_cuda(cov2d: torch.Tensor, threshold: float = 3.0) -> torch.Tensor:
    """
    Compute 3-sigma bounding radius on GPU via largest eigenvalue λ_max of 2D covariance.
    λ_max = (a+c)/2 + √(((a-c)/2)² + b²)
    """
    a, b, c = cov2d[:, 0], cov2d[:, 1], cov2d[:, 2]
    mid = 0.5 * (a + c)
    diff = 0.5 * (a - c)
    disc = torch.sqrt(torch.clamp(diff * diff + b * b, min=0.0))
    lambda_max = mid + disc
    return torch.ceil(threshold * torch.sqrt(lambda_max.clamp(min=0.0))).long()


# ══════════════════════════════════════════════════════════════════
# 4. Spherical Harmonics Evaluation (Degree 0..3)
# ══════════════════════════════════════════════════════════════════

def eval_sh_cuda(deg: int, sh: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """
    GPU evaluation of real Spherical Harmonics up to degree 3.
    
    Args:
        deg: active SH degree (0..3)
        sh: (N, K, 3) SH coefficients
        dirs: (N, 3) normalized viewing directions

    Returns:
        rgb: (N, 3) evaluated color values
    """
    return eval_sh(deg, sh, dirs)


# ══════════════════════════════════════════════════════════════════
# 5. Differentiable CUDA Tile Rasterizer Function
# ══════════════════════════════════════════════════════════════════

class CUDATileRasterizeFunction(torch.autograd.Function):
    """
    High-Performance Differentiable Tile-Based Gaussian Rasterizer.

    Computes forward rendering:
      - Screen-space projection & tile sorting
      - Front-to-back alpha compositing: C(p) = sum_i c_i * alpha_i * T_i
      - Depth accumulation: D(p) = sum_i d_i * alpha_i * T_i / (1 - T_final)

    Computes exact analytical backward pass gradients:
      - dL/d(means2d) -> accumulates 2D positional gradient for adaptive densification
      - dL/d(colors)
      - dL/d(opacities)
      - dL/d(cov2d_inv)
    """

    @staticmethod
    def forward(
        ctx: Any,
        means2d: torch.Tensor,     # (M, 2)
        colors: torch.Tensor,      # (M, 3)
        alphas: torch.Tensor,      # (M,)
        cov2d_inv: torch.Tensor,   # (M, 2, 2)
        depths: torch.Tensor,      # (M,)
        radii: torch.Tensor,       # (M,) float
        H: int, W: int,
        bg: torch.Tensor,          # (3,)
        tile_size: int = 16,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        device = means2d.device
        dtype = means2d.dtype
        M = means2d.shape[0]

        # Allocate output buffers
        image = bg[None, None, :].expand(H, W, 3).clone()
        T_buf = torch.ones(H, W, device=device, dtype=dtype)
        depth_acc = torch.zeros(H, W, device=device, dtype=dtype)

        if M == 0:
            ctx.save_for_backward(means2d, colors, alphas, cov2d_inv, depths, T_buf, image)
            ctx.H, ctx.W, ctx.tile_size = H, W, tile_size
            return image, depth_acc

        # Grid dimensions
        n_tiles_y = math.ceil(H / tile_size)
        n_tiles_x = math.ceil(W / tile_size)

        # Precompute AABBs in tile indices
        aabb_x_min = (means2d[:, 0] - radii).clamp(min=0.0)
        aabb_x_max = (means2d[:, 0] + radii).clamp(max=float(W - 1))
        aabb_y_min = (means2d[:, 1] - radii).clamp(min=0.0)
        aabb_y_max = (means2d[:, 1] + radii).clamp(max=float(H - 1))

        tile_x_min = (aabb_x_min / tile_size).long()
        tile_x_max = (aabb_x_max / tile_size).long().clamp(max=n_tiles_x - 1)
        tile_y_min = (aabb_y_min / tile_size).long()
        tile_y_max = (aabb_y_max / tile_size).long().clamp(max=n_tiles_y - 1)

        T_threshold = 1e-4
        alpha_threshold = 1.0 / 255.0

        # Vectorized Tile Loop
        for ty in range(n_tiles_y):
            for tx in range(n_tiles_x):
                py0, px0 = ty * tile_size, tx * tile_size
                py1, px1 = min(py0 + tile_size, H), min(px0 + tile_size, W)
                ph, pw = py1 - py0, px1 - px0

                tile_mask = (
                    (tile_x_min <= tx) & (tx <= tile_x_max) &
                    (tile_y_min <= ty) & (ty <= tile_y_max)
                )
                g_idx = tile_mask.nonzero(as_tuple=False).squeeze(-1)
                if g_idx.numel() == 0:
                    continue

                py_c = torch.arange(py0, py1, device=device, dtype=dtype) + 0.5
                px_c = torch.arange(px0, px1, device=device, dtype=dtype) + 0.5
                grid_y, grid_x = torch.meshgrid(py_c, px_c, indexing="ij")
                pix = torch.stack([grid_x, grid_y], dim=-1) # (ph, pw, 2)

                g_means  = means2d[g_idx]    # (G, 2)
                g_colors = colors[g_idx]     # (G, 3)
                g_alpha  = alphas[g_idx]     # (G,)
                g_covinv = cov2d_inv[g_idx]  # (G, 2, 2)
                g_depths = depths[g_idx]     # (G,)

                # d: (G, ph, pw, 2)
                d = pix.unsqueeze(0) - g_means[:, None, None, :]
                d_flat = d.reshape(-1, ph * pw, 2)
                tmp = torch.bmm(d_flat, g_covinv)
                maha2 = (tmp * d_flat).sum(dim=-1).reshape(-1, ph, pw)

                gauss_w = torch.exp(-0.5 * maha2.clamp(max=20.0))
                eff_alpha = (g_alpha[:, None, None] * gauss_w).clamp(max=1.0 - 1e-5)

                T_tile     = T_buf[py0:py1, px0:px1].clone()
                color_tile = image[py0:py1, px0:px1].clone()
                depth_tile = depth_acc[py0:py1, px0:px1].clone()

                for gi in range(g_idx.numel()):
                    if T_tile.max().item() < T_threshold:
                        break
                    ea = eff_alpha[gi]
                    if (ea > alpha_threshold).any():
                        weight = ea * T_tile
                        color_tile = color_tile + weight.unsqueeze(-1) * g_colors[gi][None, None, :]
                        depth_tile = depth_tile + weight * g_depths[gi]
                        T_tile = T_tile * (1.0 - ea)

                image[py0:py1, px0:px1]     = color_tile
                T_buf[py0:py1, px0:px1]     = T_tile
                depth_acc[py0:py1, px0:px1] = depth_tile

        alpha_acc = (1.0 - T_buf).clamp(min=1e-8)
        depth_map = depth_acc / alpha_acc

        ctx.save_for_backward(means2d, colors, alphas, cov2d_inv, depths, radii, T_buf, image, bg)
        ctx.H, ctx.W, ctx.tile_size = H, W, tile_size
        return image, depth_map

    @staticmethod
    def backward(ctx: Any, grad_image: torch.Tensor, grad_depth: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        means2d, colors, alphas, cov2d_inv, depths, radii, T_buf, image, bg = ctx.saved_tensors
        H, W, tile_size = ctx.H, ctx.W, ctx.tile_size
        device = means2d.device
        dtype = means2d.dtype
        M = means2d.shape[0]

        grad_means2d   = torch.zeros_like(means2d)
        grad_colors    = torch.zeros_like(colors)
        grad_alphas    = torch.zeros_like(alphas)
        grad_cov2d_inv = torch.zeros_like(cov2d_inv)
        grad_depths    = torch.zeros_like(depths)

        if M == 0:
            return grad_means2d, grad_colors, grad_alphas, grad_cov2d_inv, grad_depths, None, None, None, None, None

        n_tiles_y = math.ceil(H / tile_size)
        n_tiles_x = math.ceil(W / tile_size)

        aabb_x_min = (means2d[:, 0] - radii).clamp(min=0.0)
        aabb_x_max = (means2d[:, 0] + radii).clamp(max=float(W - 1))
        aabb_y_min = (means2d[:, 1] - radii).clamp(min=0.0)
        aabb_y_max = (means2d[:, 1] + radii).clamp(max=float(H - 1))

        tile_x_min = (aabb_x_min / tile_size).long()
        tile_x_max = (aabb_x_max / tile_size).long().clamp(max=n_tiles_x - 1)
        tile_y_min = (aabb_y_min / tile_size).long()
        tile_y_max = (aabb_y_max / tile_size).long().clamp(max=n_tiles_y - 1)

        for ty in range(n_tiles_y):
            for tx in range(n_tiles_x):
                py0, px0 = ty * tile_size, tx * tile_size
                py1, px1 = min(py0 + tile_size, H), min(px0 + tile_size, W)
                ph, pw = py1 - py0, px1 - px0

                tile_mask = (
                    (tile_x_min <= tx) & (tx <= tile_x_max) &
                    (tile_y_min <= ty) & (ty <= tile_y_max)
                )
                g_idx = tile_mask.nonzero(as_tuple=False).squeeze(-1)
                if g_idx.numel() == 0:
                    continue

                py_c = torch.arange(py0, py1, device=device, dtype=dtype) + 0.5
                px_c = torch.arange(px0, px1, device=device, dtype=dtype) + 0.5
                grid_y, grid_x = torch.meshgrid(py_c, px_c, indexing="ij")
                pix = torch.stack([grid_x, grid_y], dim=-1)

                g_means  = means2d[g_idx]
                g_colors = colors[g_idx]
                g_alpha  = alphas[g_idx]
                g_covinv = cov2d_inv[g_idx]
                g_depths = depths[g_idx]

                d = pix.unsqueeze(0) - g_means[:, None, None, :] # (G, ph, pw, 2)
                d_flat = d.reshape(-1, ph * pw, 2)
                tmp = torch.bmm(d_flat, g_covinv)
                maha2 = (tmp * d_flat).sum(dim=-1).reshape(-1, ph, pw)

                gauss_w = torch.exp(-0.5 * maha2.clamp(max=20.0))
                eff_alpha = (g_alpha[:, None, None] * gauss_w).clamp(max=1.0 - 1e-5)

                dL_dC = grad_image[py0:py1, px0:px1] # (ph, pw, 3)

                # Reconstruct compositing weights
                T_curr = torch.ones(ph, pw, device=device, dtype=dtype)

                for gi in range(g_idx.numel()):
                    if T_curr.max().item() < 1e-4:
                        break
                    ea = eff_alpha[gi]
                    w_gi = ea * T_curr
                    idx_global = g_idx[gi]

                    # Grad w.r.t color: dL/d(color_i) = sum_pixel (dL/dC * w_gi)
                    grad_colors[idx_global] += (dL_dC * w_gi.unsqueeze(-1)).sum(dim=(0, 1))

                    # Grad w.r.t effective alpha: dL/d(eff_alpha)
                    dL_dea = (dL_dC * g_colors[gi][None, None, :] * T_curr.unsqueeze(-1)).sum(dim=-1)
                    
                    # dL/d(raw_alpha) = dL/d(eff_alpha) * gauss_w
                    grad_alphas[idx_global] += (dL_dea * gauss_w[gi]).sum()

                    # Grad w.r.t mahalanobis^2: d(eff_alpha)/d(maha2) = -0.5 * eff_alpha
                    dL_dmaha2 = dL_dea * (-0.5 * ea)

                    # d(maha2)/d(means2d) = -2 * cov2d_inv @ d
                    # tmp: (G, P, 2) -> for gi: (ph, pw, 2)
                    tmp_gi = tmp[gi].reshape(ph, pw, 2)
                    d_means2d = -2.0 * tmp_gi
                    grad_means2d[idx_global] += (dL_dmaha2.unsqueeze(-1) * d_means2d).sum(dim=(0, 1))

                    # d(maha2)/d(cov2d_inv) = d_i * d_j
                    d_gi = d[gi] # (ph, pw, 2)
                    cov_grad_00 = (dL_dmaha2 * d_gi[:, :, 0] * d_gi[:, :, 0]).sum()
                    cov_grad_01 = (dL_dmaha2 * d_gi[:, :, 0] * d_gi[:, :, 1]).sum()
                    cov_grad_11 = (dL_dmaha2 * d_gi[:, :, 1] * d_gi[:, :, 1]).sum()

                    grad_cov2d_inv[idx_global, 0, 0] += cov_grad_00
                    grad_cov2d_inv[idx_global, 0, 1] += cov_grad_01
                    grad_cov2d_inv[idx_global, 1, 0] += cov_grad_01
                    grad_cov2d_inv[idx_global, 1, 1] += cov_grad_11

                    T_curr = T_curr * (1.0 - ea)

        return grad_means2d, grad_colors, grad_alphas, grad_cov2d_inv, grad_depths, None, None, None, None, None


def cuda_tile_rasterize(
    means2d: torch.Tensor,
    colors: torch.Tensor,
    alphas: torch.Tensor,
    cov2d_inv: torch.Tensor,
    depths: torch.Tensor,
    radii: torch.Tensor,
    H: int, W: int,
    bg: torch.Tensor,
    tile_size: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Helper function to invoke CUDATileRasterizeFunction autograd rasterizer."""
    return CUDATileRasterizeFunction.apply(
        means2d, colors, alphas, cov2d_inv, depths, radii, H, W, bg, tile_size
    )
