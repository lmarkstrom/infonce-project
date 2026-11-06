"""
Gaussian structure metrics for contrastive representations.

Implements:
  - CV (coefficient of variation) of representation norms  (Eq. 20)
  - Anderson-Darling (AD) normality test per coordinate
  - D'Agostino-Pearson (DP) normality test per coordinate
  - Alignment: mean cosine similarity of positive pairs
  - Uniformity: mean cosine similarity of negative pairs
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import normaltest  # D'Agostino-Pearson
from scipy.stats import anderson    # Anderson-Darling


# ---------------------------------------------------------------------------
# Norm concentration
# ---------------------------------------------------------------------------

def coefficient_of_variation(representations: torch.Tensor) -> float:
    """
    Coefficient of variation of representation norms (Eq. 20).

    CV = std(||z_i||) / mean(||z_i||)

    Args:
        representations: Tensor of shape (N, d), unnormalized embeddings.

    Returns:
        Scalar float, CV of norms.
    """
    norms = representations.norm(dim=-1).cpu().float()
    cv = (norms.std() / norms.mean()).item()
    return cv


def mean_norm(representations: torch.Tensor) -> float:
    """Mean representation norm."""
    return representations.norm(dim=-1).cpu().float().mean().item()


# ---------------------------------------------------------------------------
# Gaussianity tests
# ---------------------------------------------------------------------------

def anderson_darling_test(representations: torch.Tensor) -> dict:
    """
    Anderson-Darling normality test applied per coordinate.

    Returns the average AD statistic and the fraction of coordinates where
    we fail to reject normality (AD statistic < 0.752, the 5% critical value
    for the 'norm' distribution as tabulated by scipy).

    Args:
        representations: Tensor of shape (N, d).

    Returns:
        dict with keys:
          'ad_avg'   : average AD statistic across coordinates
          'ad_norm_frac' : fraction of coordinates with AD < 0.752
          'ad_stats' : array of per-coordinate AD statistics
    """
    Z = representations.cpu().float().numpy()  # (N, d)
    N, d = Z.shape

    ad_stats = np.empty(d)
    for j in range(d):
        # Use statistic only; avoid FutureWarning about method parameter.
        # The statistic itself is independent of the p-value computation method.
        result = anderson(Z[:, j], dist='norm')
        ad_stats[j] = result.statistic

    # 5% critical value for the Anderson-Darling test against normality is 0.752
    threshold = 0.752
    ad_norm_frac = float(np.mean(ad_stats < threshold))
    ad_avg = float(np.mean(ad_stats))

    return {
        'ad_avg': ad_avg,
        'ad_norm_frac': ad_norm_frac,
        'ad_stats': ad_stats,
    }


def dagostino_pearson_test(representations: torch.Tensor) -> dict:
    """
    D'Agostino-Pearson normality test applied per coordinate.

    Returns the average p-value and the fraction of coordinates where we
    fail to reject normality (p-value > 0.05).

    Args:
        representations: Tensor of shape (N, d).

    Returns:
        dict with keys:
          'dp_avg'       : average p-value across coordinates
          'dp_norm_frac' : fraction of coordinates with p > 0.05
          'dp_pvalues'   : array of per-coordinate p-values
    """
    Z = representations.cpu().float().numpy()  # (N, d)
    N, d = Z.shape

    dp_pvalues = np.empty(d)
    for j in range(d):
        _, p = normaltest(Z[:, j])
        dp_pvalues[j] = p

    threshold = 0.05
    dp_norm_frac = float(np.mean(dp_pvalues > threshold))
    dp_avg = float(np.mean(dp_pvalues))

    return {
        'dp_avg': dp_avg,
        'dp_norm_frac': dp_norm_frac,
        'dp_pvalues': dp_pvalues,
    }


def gaussian_diagnostics(representations: torch.Tensor) -> dict:
    """
    Full Gaussian diagnostics: CV, AD, DP.

    Args:
        representations: Tensor of shape (N, d), unnormalized embeddings.

    Returns:
        Combined dict with all metrics.
    """
    result = {}
    result['cv'] = coefficient_of_variation(representations)
    result['mean_norm'] = mean_norm(representations)
    result.update(anderson_darling_test(representations))
    result.update(dagostino_pearson_test(representations))
    return result


# ---------------------------------------------------------------------------
# Alignment and uniformity (Figure 2 / Appendix Figs 7, 8)
# ---------------------------------------------------------------------------

def alignment(u: torch.Tensor, v: torch.Tensor) -> float:
    """
    Mean cosine similarity of positive pairs.

    Args:
        u: (N, d) L2-normalized representations of first views.
        v: (N, d) L2-normalized representations of second views.

    Returns:
        Scalar float ∈ [-1, 1].
    """
    u_n = F.normalize(u, dim=-1)
    v_n = F.normalize(v, dim=-1)
    return (u_n * v_n).sum(dim=-1).mean().item()


def uniformity(u: torch.Tensor) -> float:
    """
    Mean cosine similarity of negative pairs (a measure of how spread the
    representations are on the sphere). Smaller → more uniform.

    Args:
        u: (N, d) L2-normalized representations.

    Returns:
        Scalar float — mean off-diagonal cosine similarity.
    """
    u_n = F.normalize(u, dim=-1)
    # Pairwise cosine similarity matrix
    sim = torch.mm(u_n, u_n.T)  # (N, N)
    N = u.shape[0]
    # Mask diagonal
    mask = ~torch.eye(N, dtype=torch.bool, device=u.device)
    return sim[mask].mean().item()
