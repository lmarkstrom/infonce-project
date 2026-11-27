"""Gaussianity diagnostics, norm concentration, alignment/uniformity, whitening, and pushforward metrics.

Implements all evaluation metrics from Section 5 and Appendix E of:
'InfoNCE Induces Gaussian Distribution' (Betser, Gofer, Levi, & Gilboa, ICLR 2026).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats

AD_THRESHOLD = 0.752
DP_PVALUE_THRESHOLD = 0.05


def to_numpy(x: Any) -> np.ndarray:
    """Convert a PyTorch tensor or array-like to a contiguous float64 NumPy array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(x, dtype=np.float64))


def l2_normalize_numpy(z: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize rows to S^{d-1}, assigning c_0 = e_1 for zero-norm rows (Section 3)."""
    z = to_numpy(z)
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    is_zero = norms <= eps
    safe_norms = np.where(is_zero, 1.0, norms)
    u = z / safe_norms
    if np.any(is_zero):
        c0 = np.zeros_like(u)
        c0[:, 0] = 1.0
        u = np.where(is_zero, c0, u)
    return u


def compute_norm_stats(embeddings: np.ndarray) -> dict[str, Any]:
    """Compute representation norm mean, std, and Coefficient of Variation (Eq. 20)."""
    z = to_numpy(embeddings)
    norms = np.linalg.norm(z, axis=1)
    norm_mean = float(np.mean(norms))
    norm_std = float(np.std(norms, ddof=0))
    cv = float(norm_std / norm_mean) if norm_mean > 1e-12 else float("inf")
    return {
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "cv": cv,
        "norms": norms,
    }


def compute_gaussianity_diagnostics(
    embeddings: np.ndarray,
    ad_threshold: float = AD_THRESHOLD,
    dp_threshold: float = DP_PVALUE_THRESHOLD,
) -> dict[str, Any]:
    """Evaluate radial thin-shell concentration (CV) and coordinate-wise normality (AD & DP).

    Args:
        embeddings: Array of shape (N, d).
        ad_threshold: Anderson-Darling threshold (paper: 0.752).
        dp_threshold: D'Agostino-Pearson p-value threshold (paper: 0.05).

    Returns:
        Dictionary containing CV, norm mean/std, AD average & passing fraction,
        DP average p-value & passing fraction, and per-coordinate arrays.
    """
    z = to_numpy(embeddings)
    if z.ndim != 2:
        raise ValueError(f"Expected 2D embeddings (N, d), got shape {z.shape}")
    n_samples, dim = z.shape
    if n_samples < 8:
        raise ValueError(f"D'Agostino-Pearson test requires at least 8 samples, got {n_samples}")

    norm_info = compute_norm_stats(z)

    ad_stats = np.empty(dim, dtype=np.float64)
    ad_pass = np.zeros(dim, dtype=bool)
    dp_stats = np.empty(dim, dtype=np.float64)
    dp_pvals = np.empty(dim, dtype=np.float64)
    dp_pass = np.zeros(dim, dtype=bool)
    n_constant_dims = 0

    for j in range(dim):
        col = z[:, j]
        col_std = float(np.std(col))
        if not np.isfinite(col_std) or col_std < 1e-12:
            n_constant_dims += 1
            ad_stats[j] = np.nan
            ad_pass[j] = False
            dp_stats[j] = np.nan
            dp_pvals[j] = 0.0
            dp_pass[j] = False
            continue

        try:
            ad_res = stats.anderson(col, dist="norm", method="interpolate")
        except TypeError:
            ad_res = stats.anderson(col, dist="norm")
        ad_val = float(ad_res.statistic)
        ad_stats[j] = ad_val
        ad_pass[j] = bool(np.isfinite(ad_val) and ad_val < ad_threshold)

        dp_stat, dp_p = stats.normaltest(col)
        dp_stats[j] = float(dp_stat)
        dp_pvals[j] = float(dp_p) if np.isfinite(dp_p) else 0.0
        dp_pass[j] = bool(np.isfinite(dp_p) and dp_p > dp_threshold)

    finite_ad = ad_stats[np.isfinite(ad_stats)]
    ad_avg = float(np.mean(finite_ad)) if finite_ad.size > 0 else float("inf")
    ad_norm_feat_pct = float(100.0 * np.mean(ad_pass))
    dp_avg = float(np.mean(dp_pvals))
    dp_norm_feat_pct = float(100.0 * np.mean(dp_pass))

    # Overall qualitative Gaussian check consistent with Tables 1 & 2
    is_gaussian = bool(
        ad_avg < ad_threshold
        and dp_avg > dp_threshold
        and ad_norm_feat_pct >= 80.0
        and dp_norm_feat_pct >= 80.0
    )

    return {
        "n_samples": int(n_samples),
        "dim": int(dim),
        "n_constant_dims": int(n_constant_dims),
        "norm_mean": norm_info["norm_mean"],
        "norm_std": norm_info["norm_std"],
        "cv": norm_info["cv"],
        "ad_avg": ad_avg,
        "ad_norm_feat_pct": ad_norm_feat_pct,
        "dp_avg": dp_avg,
        "dp_norm_feat_pct": dp_norm_feat_pct,
        "is_gaussian": is_gaussian,
    }


def evaluate_embeddings_both(
    embeddings: np.ndarray,
    ad_threshold: float = AD_THRESHOLD,
    dp_threshold: float = DP_PVALUE_THRESHOLD,
) -> dict[str, dict[str, Any]]:
    """Evaluate Gaussianity diagnostics for both unnormalized and unit-normalized embeddings."""
    z_unnorm = to_numpy(embeddings)
    z_norm = l2_normalize_numpy(z_unnorm)
    return {
        "unnormalized": compute_gaussianity_diagnostics(
            z_unnorm, ad_threshold=ad_threshold, dp_threshold=dp_threshold
        ),
        "normalized": compute_gaussianity_diagnostics(
            z_norm, ad_threshold=ad_threshold, dp_threshold=dp_threshold
        ),
    }


def compute_negative_cosines(
    embeddings: np.ndarray,
    max_pairs: int = 200_000,
    seed: int = 42,
) -> np.ndarray:
    """Compute cosine similarities between distinct (negative) embedding pairs.

    For small N where N*(N-1)/2 <= max_pairs, returns all unique upper-triangular
    negative pairs exactly. For larger N, randomly samples `max_pairs` distinct
    index pairs (i != j) without self-pairs.
    """
    u = l2_normalize_numpy(embeddings)
    n = u.shape[0]
    total_unique = n * (n - 1) // 2
    if total_unique <= max_pairs:
        sim = u @ u.T
        iu, ju = np.triu_indices(n, k=1)
        return sim[iu, ju].astype(np.float32)

    rng = np.random.default_rng(seed)
    idx_i = rng.integers(0, n, size=max_pairs)
    idx_j = rng.integers(0, n - 1, size=max_pairs)
    idx_j = np.where(idx_j >= idx_i, idx_j + 1, idx_j)
    return np.sum(u[idx_i] * u[idx_j], axis=1).astype(np.float32)


def compute_alignment_uniformity_stats(
    z1: np.ndarray,
    z2: np.ndarray,
    temperature: float = 0.1,
    max_neg_pairs: int = 200_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute positive-pair (alignment) and negative-pair (uniformity) cosine distributions.

    Also computes the population InfoNCE alignment and uniformity potential (Eq. 4).
    """
    u = l2_normalize_numpy(z1)
    v = l2_normalize_numpy(z2)
    n = u.shape[0]

    pos_cos = np.sum(u * v, axis=1).astype(np.float32)

    # Sample negative pairs across (u_i, v_j) with i != j
    total_neg = n * (n - 1)
    if total_neg <= max_neg_pairs:
        sim = u @ v.T
        mask = ~np.eye(n, dtype=bool)
        neg_cos = sim[mask].astype(np.float32)
    else:
        rng = np.random.default_rng(seed)
        idx_i = rng.integers(0, n, size=max_neg_pairs)
        idx_j = rng.integers(0, n - 1, size=max_neg_pairs)
        idx_j = np.where(idx_j >= idx_i, idx_j + 1, idx_j)
        neg_cos = np.sum(u[idx_i] * v[idx_j], axis=1).astype(np.float32)

    # Estimate population uniformity potential Phi(mu) on a manageable block if N is large
    alpha = 1.0 / temperature
    sub_n = min(n, 2000)
    u_sub = u[:sub_n]
    sim_sub = (u_sub @ u_sub.T) * alpha
    np.fill_diagonal(sim_sub, -np.inf)
    max_s = np.max(sim_sub, axis=1, keepdims=True)
    log_mean_exp = max_s[:, 0] + np.log(
        np.mean(np.exp(sim_sub - max_s), axis=1) * (sub_n / max(sub_n - 1, 1))
    )
    uniformity_phi = float(np.mean(log_mean_exp))
    alignment_mean = float(np.mean(pos_cos))

    return {
        "alignment_mean": alignment_mean,
        "alignment_std": float(np.std(pos_cos)),
        "uniformity_neg_cos_mean": float(np.mean(neg_cos)),
        "uniformity_neg_cos_std": float(np.std(neg_cos)),
        "uniformity_phi": uniformity_phi,
        "population_infonce": float(-alpha * alignment_mean + uniformity_phi),
        "pos_cos": pos_cos,
        "neg_cos": neg_cos,
    }


def zca_whiten(embeddings: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Apply ZCA whitening (zero mean, identity covariance) to embeddings.

    Used in Appendix E.2 (Figures 9 and 10) to evaluate post-hoc whitening
    on pretrained representations.
    """
    z = to_numpy(embeddings)
    mean = np.mean(z, axis=0, keepdims=True)
    zc = z - mean
    cov = (zc.T @ zc) / max(z.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals_clipped = np.maximum(eigvals, eps)
    inv_sqrt = (eigvecs / np.sqrt(eigvals_clipped)) @ eigvecs.T
    return zc @ inv_sqrt


def laplace_log_likelihood(x: np.ndarray, loc: float = 0.0, scale: float = 1.0) -> np.ndarray:
    """Exact log-density of i.i.d. Laplace(loc, scale) vectors."""
    arr = to_numpy(x)
    const = -math.log(2.0 * scale)
    return np.sum(const - np.abs(arr - loc) / scale, axis=1)


def fit_gaussian_log_likelihood(
    z_fit: np.ndarray,
    z_eval: np.ndarray,
    reg: float = 1e-5,
) -> np.ndarray:
    """Fit a multivariate Gaussian N(mu, Sigma) on z_fit and compute log-density on z_eval."""
    zf = to_numpy(z_fit)
    ze = to_numpy(z_eval)
    d = zf.shape[1]
    mu = np.mean(zf, axis=0)
    zc = zf - mu
    cov = (zc.T @ zc) / max(zf.shape[0] - 1, 1)
    cov = cov + reg * np.eye(d, dtype=np.float64)

    chol = np.linalg.cholesky(cov)
    log_det = 2.0 * float(np.sum(np.log(np.diag(chol))))
    diff = (ze - mu).T  # (d, N)
    solved = np.linalg.solve(chol, diff)
    quad = np.sum(solved * solved, axis=0)
    return -0.5 * (d * math.log(2.0 * math.pi) + log_det + quad)


def compute_pushforward_likelihood_correlation(
    x_test: np.ndarray,
    z_train: np.ndarray,
    z_test: np.ndarray,
    reg: float = 1e-5,
) -> dict[str, Any]:
    """Compute pushforward log-likelihood Pearson correlation (Appendix E.2, Fig. 11)."""
    log_p_data = laplace_log_likelihood(x_test, loc=0.0, scale=1.0)
    log_p_emb = fit_gaussian_log_likelihood(z_train, z_test, reg=reg)
    r_val, p_val = stats.pearsonr(log_p_data, log_p_emb)
    slope, intercept = np.polyfit(log_p_emb, log_p_data, deg=1)
    return {
        "pearson_r": float(r_val),
        "pearson_p": float(p_val),
        "fit_slope": float(slope),
        "fit_intercept": float(intercept),
        "log_p_data": log_p_data.astype(np.float32),
        "log_p_emb": log_p_emb.astype(np.float32),
    }

