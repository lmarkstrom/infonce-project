"""
Shared evaluation utilities and final report generation.

Provides:
  - evaluate_model(): compute all Gaussian diagnostics for any encoder + loader
  - generate_full_report(): run quick sanity checks on all three experiments
  - whitening(): apply ZCA whitening to a representation matrix
"""

from __future__ import annotations

import os
import json
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import gaussian_diagnostics, alignment, uniformity, coefficient_of_variation


RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# Whitening
# ---------------------------------------------------------------------------

def whiten(Z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    ZCA whitening: decorrelate and rescale to unit variance per dimension.

    Computed as Z_w = (Z - mean) @ (Cov + eps*I)^{-1/2}

    Args:
        Z: Tensor of shape (N, d), unnormalized representations.
        eps: Regularization for covariance inversion.

    Returns:
        Z_w: Whitened Tensor of shape (N, d).
    """
    Z = Z.float()
    Z_centered = Z - Z.mean(dim=0, keepdim=True)
    cov = Z_centered.T @ Z_centered / (Z.shape[0] - 1)
    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    # Whitening matrix: W = V @ diag(1/sqrt(lambda + eps)) @ V^T
    W = eigenvectors @ torch.diag(1.0 / torch.sqrt(eigenvalues + eps)) @ eigenvectors.T
    return Z_centered @ W


# ---------------------------------------------------------------------------
# General model evaluator
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    repr_fn=None,
) -> dict:
    """
    Extract representations from a model and compute Gaussian diagnostics.

    Args:
        model: PyTorch model.
        loader: DataLoader yielding (x, ...) batches.
        repr_fn: Optional callable model → representations. If None, uses
                 model(x) directly.

    Returns:
        dict with all Gaussian diagnostic metrics.
    """
    model.eval()
    all_z = []
    for batch in loader:
        x = batch[0].to(DEVICE)
        if repr_fn is not None:
            z = repr_fn(model, x)
        else:
            z = model(x)
        all_z.append(z.cpu())
    Z = torch.cat(all_z, dim=0)
    return gaussian_diagnostics(Z)


# ---------------------------------------------------------------------------
# Norm histogram plot (Figure 3 supplement)
# ---------------------------------------------------------------------------

def plot_norm_histogram(Z: torch.Tensor, title: str, save_path: str):
    """Plot histogram of representation norms."""
    norms = Z.norm(dim=-1).cpu().numpy()
    cv = norms.std() / norms.mean()

    plt.figure(figsize=(5, 4))
    plt.hist(norms, bins=50, density=True, alpha=0.7, color='steelblue')
    plt.xlabel('||z||')
    plt.ylabel('Density')
    plt.title(f'{title}\nCV={cv:.3f}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Cosine similarity histogram (Figures 7, 8, 9, 10)
# ---------------------------------------------------------------------------

def plot_cosine_similarity_histograms(
    Z: torch.Tensor,
    title: str,
    save_path: str,
    n_samples: int = 2000,
):
    """
    Plot histograms of pairwise cosine similarities.

    Shows the distribution of:
      - Positive pair similarities (diagonal of sim matrix, if available)
      - Negative pair similarities (off-diagonal)

    Args:
        Z: Tensor (N, d), representations for one view.
        title: Plot title.
        save_path: Where to save.
        n_samples: Subsample for efficiency.
    """
    Z_norm = F.normalize(Z, dim=-1).cpu()
    if Z.shape[0] > n_samples:
        idx = torch.randperm(Z.shape[0])[:n_samples]
        Z_norm = Z_norm[idx]

    sim_matrix = (Z_norm @ Z_norm.T).numpy()
    N = Z_norm.shape[0]
    # Off-diagonal = negative pair similarities
    neg_sims = sim_matrix[~np.eye(N, dtype=bool)].flatten()

    plt.figure(figsize=(5, 4))
    plt.hist(neg_sims, bins=100, density=True, alpha=0.7, color='tomato', label='Negatives')
    plt.xlabel('Cosine similarity')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Full summary report
# ---------------------------------------------------------------------------

def generate_full_report():
    """
    Load saved results and generate a comprehensive summary table
    comparing all experimental settings.
    """
    report_path = os.path.join(RESULTS_DIR, 'full_report.txt')

    sections = []

    # Try to load saved results
    laplace_path = os.path.join(RESULTS_DIR, 'laplace_grid_results.json')
    if os.path.exists(laplace_path):
        with open(laplace_path) as f:
            laplace_results = json.load(f)
        # Pick best config
        best = max(laplace_results, key=lambda r: r.get('ad_norm_frac', 0))
        sections.append(
            f"Laplace (best d={best['out_dim']}, N={best['batch_size']}): "
            f"CV={best['cv']:.3f}, AD_avg={best['ad_avg']:.3f}, "
            f"AD%={best['ad_norm_frac']*100:.1f}%, "
            f"DP_avg={best['dp_avg']:.3f}, DP%={best['dp_norm_frac']*100:.1f}%"
        )

    for fname, label in [
        ('cifar10_resnet_contrastive_diag.json', 'CIFAR-10 Contrastive (ResNet-18)'),
        ('cifar10_resnet_supervised_diag.json', 'CIFAR-10 Supervised (ResNet-18)'),
    ]:
        path = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            sections.append(
                f"{label}: CV={d.get('cv', 'N/A'):.3f}, "
                f"AD_avg={d.get('ad_avg', 'N/A'):.3f}, "
                f"AD%={d.get('ad_norm_frac', 0)*100:.1f}%, "
                f"DP_avg={d.get('dp_avg', 'N/A'):.3f}, "
                f"DP%={d.get('dp_norm_frac', 0)*100:.1f}%"
            )

    report = '\n'.join(['=== Full Experiment Report ==='] + sections)
    print(report)
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")


if __name__ == '__main__':
    generate_full_report()
