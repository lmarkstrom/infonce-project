"""
InfoNCE loss implementation.

InfoNCE (Oord et al., 2018) as used in the paper:
  L_InfoNCE = -1/N * sum_i log( exp(u_i·v_i/τ) / sum_j exp(u_i·v_j/τ) )

where u_i = f_hat(x_i), v_j = f_hat(y_j) are L2-normalized representations,
and τ is the temperature parameter.
"""

import torch
import torch.nn.functional as F


def infonce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    Symmetric InfoNCE loss operating on L2-normalized representations.

    Args:
        z1: Tensor of shape (N, d), first views (will be L2-normalized internally).
        z2: Tensor of shape (N, d), second views (will be L2-normalized internally).
        temperature: Temperature τ > 0.

    Returns:
        Scalar loss value.
    """
    # L2-normalize to get unit representations u_i, v_i  (Eq. 2 in paper)
    u = F.normalize(z1, dim=-1)   # (N, d)
    v = F.normalize(z2, dim=-1)   # (N, d)

    N = u.shape[0]

    # Similarity matrix: logits[i, j] = u_i · v_j / τ
    # Rows: anchors u_i, Columns: candidates v_j
    logits = torch.mm(u, v.T) / temperature  # (N, N)

    # Positive pairs are on the diagonal (i == j)
    labels = torch.arange(N, device=u.device)

    # Cross-entropy over rows (each u_i must rank its v_i highest)
    loss_rows = F.cross_entropy(logits, labels)

    # Cross-entropy over columns (symmetric: each v_i ranks its u_i highest)
    loss_cols = F.cross_entropy(logits.T, labels)

    return (loss_rows + loss_cols) / 2.0
