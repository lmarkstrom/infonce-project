"""Contrastive loss functions and population alignment/uniformity objectives.

Implements:
- Empirical InfoNCE loss (Section 3, Eq. 2)
- Symmetric InfoNCE and SimCLR NT-Xent loss (for SimCLR ResNet-18 protocol)
- Population alignment and uniformity potential diagnostics (Section 3, Eq. 4)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_l2_normalize(z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize representations to the unit sphere S^{d-1} as in Section 3.

    For any sample with ||z|| <= eps, assigns the fixed unit vector
    c_0 = (1, 0, ..., 0) in S^{d-1} as specified in the paper.
    """
    norms = torch.linalg.vector_norm(z, ord=2, dim=-1, keepdim=True)
    is_zero = norms <= eps
    safe_norms = torch.where(is_zero, torch.ones_like(norms), norms)
    u = z / safe_norms
    if torch.any(is_zero):
        c0 = torch.zeros_like(u)
        c0[..., 0] = 1.0
        u = torch.where(is_zero, c0, u)
    return u


class InfoNCELoss(nn.Module):
    """Empirical InfoNCE loss from Section 3, Eq. (2).

    Given two batches of representations (z1, z2) corresponding to N positive pairs
    {(x_i, y_i)}_{i=1}^N:
        u_i = f_hat(x_i),  v_i = f_hat(y_i)
        L_InfoNCE = -(1/N) sum_{i=1}^N log [ exp(<u_i, v_i>/tau) / sum_{j=1}^N exp(<u_i, v_j>/tau) ]

    Args:
        temperature: Positive temperature parameter tau (paper default: 0.1).
        symmetric: If True, averages L(u, v) and L(v, u).
    """

    def __init__(self, temperature: float = 0.1, symmetric: bool = False) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.temperature = float(temperature)
        self.symmetric = bool(symmetric)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        if z1.shape != z2.shape:
            raise ValueError(f"z1 and z2 must have identical shapes, got {z1.shape} vs {z2.shape}")
        u = safe_l2_normalize(z1)
        v = safe_l2_normalize(z2)
        batch_size = u.shape[0]
        labels = torch.arange(batch_size, device=u.device, dtype=torch.long)

        logits_uv = torch.matmul(u, v.T) / self.temperature
        loss_uv = F.cross_entropy(logits_uv, labels)
        if not self.symmetric:
            return loss_uv

        logits_vu = logits_uv.T
        loss_vu = F.cross_entropy(logits_vu, labels)
        return 0.5 * (loss_uv + loss_vu)


class SimCLRNTXentLoss(nn.Module):
    """Standard SimCLR NT-Xent loss (Chen et al., 2020) over 2N augmented views.

    For N positive pairs (z1, z2), concatenates [u; v] of size 2N, masks out
    self-similarity (diagonal), and computes cross-entropy against the paired view
    with temperature tau.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.temperature = float(temperature)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        if z1.shape != z2.shape:
            raise ValueError(f"z1 and z2 must have identical shapes, got {z1.shape} vs {z2.shape}")
        batch_size = z1.shape[0]
        u = safe_l2_normalize(z1)
        v = safe_l2_normalize(z2)
        reps = torch.cat([u, v], dim=0)  # (2N, d)

        logits = torch.matmul(reps, reps.T) / self.temperature  # (2N, 2N)
        mask = torch.eye(2 * batch_size, device=reps.device, dtype=torch.bool)
        logits = logits.masked_fill(mask, float("-inf"))

        targets = torch.cat(
            [
                torch.arange(batch_size, 2 * batch_size, device=reps.device, dtype=torch.long),
                torch.arange(0, batch_size, device=reps.device, dtype=torch.long),
            ],
            dim=0,
        )
        return F.cross_entropy(logits, targets)


@torch.no_grad()
def compute_population_infonce_terms(
    u: torch.Tensor,
    v: torch.Tensor,
    temperature: float = 0.1,
) -> dict[str, float]:
    """Estimate the empirical alignment and uniformity potential terms from Eq. (4).

    Eq. (4): L(mu, pi) = -alpha * E_{(u,v)~pi}[u . v] + Phi(mu),
    where alpha = 1 / tau and Phi(mu) = E_{u~mu} log E_{v~mu} exp(alpha * u . v).
    """
    u = safe_l2_normalize(u)
    v = safe_l2_normalize(v)
    alpha = 1.0 / temperature
    n = u.shape[0]

    alignment = (u * v).sum(dim=-1).mean().item()

    sim = torch.matmul(u, u.T) * alpha
    mask = torch.eye(n, device=u.device, dtype=torch.bool)
    sim_off = sim.masked_fill(mask, float("-inf"))
    # E_u [ log ( (1/(N-1)) sum_{j != i} exp(alpha * u_i . u_j) ) ]
    uniformity_phi = (torch.logsumexp(sim_off, dim=1) - math.log(max(n - 1, 1))).mean().item()
    pop_loss = -alpha * alignment + uniformity_phi

    return {
        "alignment_cosine": float(alignment),
        "uniformity_phi": float(uniformity_phi),
        "population_infonce": float(pop_loss),
    }

