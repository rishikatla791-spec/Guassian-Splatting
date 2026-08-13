"""
[EXPERIMENTAL] Temporal Gaussian Evolution for dynamic scene support.

Models per-Gaussian velocity and acceleration fields to simulate
smooth temporal motion. Useful for dynamic scenes captured from video.

Physics model:
  Position update (Euler integration):
    x(t+dt) = x(t) + v·dt + 0.5·a·dt²

  Velocity update:
    v(t+dt) = v(t) + a·dt

  Spring model (optional):
    For two Gaussians i, j with spring constant k and rest length L₀:
      F_ij = k · (||xᵢ - xⱼ|| - L₀) · (xⱼ - xᵢ) / ||xᵢ - xⱼ||
      aᵢ   += F_ij / mᵢ   (mᵢ ∝ volume of Gaussian i)

Trajectory smoothness loss:
  L_smooth = (1/N) Σᵢ ||xᵢ(t) - 2xᵢ(t-1) + xᵢ(t-2)||²
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import List, Optional


class TemporalGaussianEvolution:
    """
    Adds velocity/acceleration dynamics to a GaussianModel for dynamic scenes.

    Attributes:
        v_xyz: (N, 3) velocity field
        a_xyz: (N, 3) acceleration field
    """

    def __init__(self, gaussians):
        """
        Args:
            gaussians: GaussianModel instance
        """
        N = gaussians.num_gaussians
        device = gaussians.get_xyz.device

        self.gaussians = gaussians
        self.v_xyz = torch.zeros(N, 3, device=device)   # velocity
        self.a_xyz = torch.zeros(N, 3, device=device)   # acceleration

        # Trajectory history for smoothness loss
        self._history: List[torch.Tensor] = []

    # -----------------------------------------------------------------------
    # Position evolution
    # -----------------------------------------------------------------------

    def evolve(self, dt: float = 1.0 / 30.0) -> None:
        """
        Advance all Gaussian positions by one timestep.

        Uses second-order Verlet/Euler integration:
            x(t+dt) = x(t) + v(t)·dt + ½·a(t)·dt²
            v(t+dt) = v(t) + a(t)·dt

        Args:
            dt: timestep in seconds (default 1/30 for 30 FPS)
        """
        # Store current position in history (detached, no grad)
        self._history.append(self.gaussians.get_xyz.detach().clone())
        if len(self._history) > 3:
            self._history.pop(0)

        with torch.no_grad():
            new_xyz = (self.gaussians._xyz.data
                       + self.v_xyz * dt
                       + 0.5 * self.a_xyz * dt ** 2)
            self.gaussians._xyz.data = new_xyz
            self.v_xyz = self.v_xyz + self.a_xyz * dt

    def apply_damping(self, damping: float = 0.99) -> None:
        """Apply velocity damping (friction): v ← v · damping."""
        self.v_xyz.mul_(damping)

    # -----------------------------------------------------------------------
    # Spring model
    # -----------------------------------------------------------------------

    def apply_spring_forces(
        self,
        neighbor_indices: torch.Tensor,
        spring_k: float = 1.0,
        rest_lengths: Optional[torch.Tensor] = None,
        mass_scale: float = 1.0,
    ) -> None:
        """
        Apply Hookean spring forces between Gaussian pairs.

        For each pair (i, neighbor_indices[i, j]):
            d_ij = x_j - x_i
            F_i += spring_k · (||d_ij|| - L₀_ij) · d_ij / ||d_ij||

        Args:
            neighbor_indices: (N, K) indices of K nearest neighbors per Gaussian
            spring_k:         spring constant
            rest_lengths:     (N, K) rest lengths. If None, computed from initial positions.
            mass_scale:       mass scaling (acceleration = force / mass_scale)
        """
        xyz = self.gaussians.get_xyz.detach()  # (N, 3)
        N, K = neighbor_indices.shape

        # d_ij = x_j - x_i: (N, K, 3)
        x_j = xyz[neighbor_indices.reshape(-1)].reshape(N, K, 3)
        d   = x_j - xyz.unsqueeze(1)  # (N, K, 3)

        dists = d.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (N, K, 1)

        if rest_lengths is None:
            rest_lengths = dists.squeeze(-1).detach()  # use current as rest

        stretch = (dists.squeeze(-1) - rest_lengths).unsqueeze(-1)  # (N, K, 1)
        forces  = spring_k * stretch * (d / dists)  # (N, K, 3)
        net_force = forces.sum(dim=1)  # (N, 3)

        self.a_xyz = self.a_xyz + net_force / mass_scale

    # -----------------------------------------------------------------------
    # Smoothness loss
    # -----------------------------------------------------------------------

    def compute_flow_loss(
        self,
        prev_gaussians_xyz: torch.Tensor,
        next_gaussians_xyz: torch.Tensor,
    ) -> torch.Tensor:
        """
        L2 trajectory smoothness loss.

        Penalizes sudden changes in velocity (second-order smoothness):
            L = ||x(t) - 2·x(t-1) + x(t-2)||²

        Args:
            prev_gaussians_xyz: (N, 3) positions at t-1
            next_gaussians_xyz: (N, 3) positions at t+1 (predicted)

        Returns:
            scalar loss tensor
        """
        current = self.gaussians.get_xyz
        accel = next_gaussians_xyz - 2.0 * current + prev_gaussians_xyz
        return (accel ** 2).mean()

    def compute_velocity_regularization(self) -> torch.Tensor:
        """Penalize excessively large velocities."""
        return (self.v_xyz ** 2).mean()

    # -----------------------------------------------------------------------
    # Tracking
    # -----------------------------------------------------------------------

    def track_gaussian(self, idx: int, n_frames: int) -> List[torch.Tensor]:
        """
        Simulate and record position trajectory of a single Gaussian.

        Args:
            idx:      Gaussian index
            n_frames: number of frames to simulate

        Returns:
            list of (3,) position tensors
        """
        # Save state
        original_xyz = self.gaussians._xyz.data.clone()
        original_v   = self.v_xyz.clone()
        original_a   = self.a_xyz.clone()

        positions = [original_xyz[idx].clone()]
        for _ in range(n_frames - 1):
            self.evolve(dt=1.0 / 30.0)
            positions.append(self.gaussians._xyz.data[idx].clone())

        # Restore state
        self.gaussians._xyz.data = original_xyz
        self.v_xyz = original_v
        self.a_xyz = original_a

        return positions
