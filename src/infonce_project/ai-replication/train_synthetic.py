"""
Synthetic data experiments (Section 5, Figure 2, Figure 3, Table 1).

Covers three settings:
  1. Laplace(0,1) inputs — linear encoder, varying d and batch size
  2. Gaussian mixture inputs — linear encoder, d=256
  3. Binary sparse inputs — linear encoder, d=256

Outputs:
  - results/synthetic_laplace_table1.txt   — Table 1 (Laplace row)
  - results/synthetic_gmm_table1.txt       — Table 1 (GMM row)
  - results/synthetic_binary_table1.txt    — Table 1 (Binary E0/E50/E100 rows)
  - results/fig2_alignment_uniformity.png  — Figure 2
  - results/fig3_norm_stats.png            — Figure 3
"""

from __future__ import annotations

import os
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datasets import LaplaceDataset, GaussianMixtureDataset, BinaryDataset
from augmentations import (
    SyntheticAugmentation,
    BinaryAugmentation,
    GMMComponentAugmentation,
    ContrastiveDatasetWrapper,
    GMMContrastiveWrapper,
)
from models import LinearEncoder
from infonce_loss import infonce_loss
from utils import (
    gaussian_diagnostics,
    alignment,
    uniformity,
    coefficient_of_variation,
)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    temperature: float,
) -> float:
    """Train for one epoch; return mean loss."""
    model.train()
    total_loss = 0.0
    for v1, v2 in loader:
        v1, v2 = v1.to(DEVICE), v2.to(DEVICE)
        z1 = model(v1)
        z2 = model(v2)
        loss = infonce_loss(z1, z2, temperature=temperature)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def extract_representations(model: nn.Module, loader: DataLoader) -> torch.Tensor:
    """Extract unnormalized representations from a dataloader."""
    model.eval()
    all_z = []
    for batch in loader:
        # Support (v1, v2) or (x, label)
        x = batch[0].to(DEVICE)
        z = model(x)
        all_z.append(z.cpu())
    return torch.cat(all_z, dim=0)


@torch.no_grad()
def collect_alignment_uniformity(
    model: nn.Module, loader: DataLoader
) -> tuple[float, float]:
    """Compute alignment and uniformity from paired batch."""
    model.eval()
    all_u, all_v = [], []
    for v1, v2 in loader:
        v1, v2 = v1.to(DEVICE), v2.to(DEVICE)
        u = torch.nn.functional.normalize(model(v1), dim=-1)
        v = torch.nn.functional.normalize(model(v2), dim=-1)
        all_u.append(u.cpu())
        all_v.append(v.cpu())
    U = torch.cat(all_u, dim=0)
    V = torch.cat(all_v, dim=0)
    aln = alignment(U, V)
    uni = uniformity(U)
    return aln, uni


# ---------------------------------------------------------------------------
# Experiment 1: Laplace data, varying d and batch size
# ---------------------------------------------------------------------------

def run_laplace_experiment(
    out_dim: int = 128,
    batch_size: int = 64,
    n_train: int = 20_000,
    n_test: int = 5_000,
    data_dim: int = 1024,
    n_epochs: int = 150,
    temperature: float = 0.1,
    A: float = 0.6,
    lr: float = 1e-3,
    seed: int = 42,
) -> dict:
    """
    Train a linear encoder on Laplace data with InfoNCE.

    Returns diagnostics dict.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Datasets
    full_dataset = LaplaceDataset(n_samples=n_train + n_test, dim=data_dim, seed=seed)
    train_ds, test_ds = random_split(
        full_dataset,
        [n_train, n_test],
        generator=torch.Generator().manual_seed(seed),
    )

    aug = SyntheticAugmentation(A=A)
    train_wrapped = ContrastiveDatasetWrapper(train_ds, aug)
    test_wrapped = ContrastiveDatasetWrapper(test_ds, aug)

    train_loader = DataLoader(train_wrapped, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    # For diagnostics, use a plain (non-paired) loader on test set
    test_loader_plain = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
    test_loader_paired = DataLoader(test_wrapped, batch_size=256, shuffle=False,
                                    num_workers=0)

    # Model
    model = LinearEncoder(in_dim=data_dim, out_dim=out_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        loss = train_one_epoch(model, train_loader, optimizer, temperature)
        if (epoch + 1) % 50 == 0:
            print(f"  [Laplace d={out_dim} N={batch_size}] epoch {epoch+1}/{n_epochs} "
                  f"loss={loss:.4f}")

    # Evaluate
    Z = extract_representations(model, test_loader_plain)
    diag = gaussian_diagnostics(Z)
    aln, uni = collect_alignment_uniformity(model, test_loader_paired)
    diag['alignment'] = aln
    diag['uniformity'] = uni
    diag['out_dim'] = out_dim
    diag['batch_size'] = batch_size
    return diag


def run_laplace_grid(
    dims: list[int] = [32, 64, 128, 256],
    batch_sizes: list[int] = [8, 16, 32, 48, 64, 96, 128],
    n_epochs: int = 150,
) -> list[dict]:
    """
    Run the Laplace experiment across all (dim, batch_size) combinations.
    Returns a list of results dicts.
    """
    results = []
    for d in dims:
        for N in batch_sizes:
            print(f"\n=== Laplace: d={d}, N={N} ===")
            res = run_laplace_experiment(out_dim=d, batch_size=N, n_epochs=n_epochs)
            results.append(res)
            print(f"  CV={res['cv']:.3f}, AD_avg={res['ad_avg']:.3f}, "
                  f"AD_norm%={res['ad_norm_frac']*100:.1f}%, "
                  f"DP_avg={res['dp_avg']:.3f}, DP_norm%={res['dp_norm_frac']*100:.1f}%, "
                  f"align={res['alignment']:.3f}, unif={res['uniformity']:.3f}")
    return results


# ---------------------------------------------------------------------------
# Experiment 2: Gaussian Mixture data
# ---------------------------------------------------------------------------

def run_gmm_experiment(
    out_dim: int = 256,
    batch_size: int = 64,
    n_samples: int = 10_000,
    data_dim: int = 1024,
    n_epochs: int = 100,
    temperature: float = 0.1,
    lr: float = 1e-3,
    seed: int = 42,
) -> dict:
    """Train a linear encoder on GMM data with InfoNCE."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    gmm_dataset = GaussianMixtureDataset(n_samples=n_samples, dim=data_dim,
                                         n_components=25, seed=seed)
    aug = GMMComponentAugmentation(means=gmm_dataset.means)
    train_wrapped = GMMContrastiveWrapper(gmm_dataset, aug)

    train_loader = DataLoader(train_wrapped, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    plain_loader = DataLoader(gmm_dataset, batch_size=256, shuffle=False, num_workers=0)
    paired_loader = DataLoader(train_wrapped, batch_size=256, shuffle=False, num_workers=0)

    model = LinearEncoder(in_dim=data_dim, out_dim=out_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        loss = train_one_epoch(model, train_loader, optimizer, temperature)
        if (epoch + 1) % 25 == 0:
            print(f"  [GMM d={out_dim}] epoch {epoch+1}/{n_epochs} loss={loss:.4f}")

    Z = extract_representations(model, plain_loader)
    diag = gaussian_diagnostics(Z)
    aln, uni = collect_alignment_uniformity(model, paired_loader)
    diag['alignment'] = aln
    diag['uniformity'] = uni
    return diag


# ---------------------------------------------------------------------------
# Experiment 3: Binary sparse data
# ---------------------------------------------------------------------------

def run_binary_experiment(
    out_dim: int = 256,
    batch_size: int = 64,
    n_samples: int = 10_000,
    data_dim: int = 1024,
    n_epochs: int = 100,
    eval_every: int = 10,
    temperature: float = 0.1,
    lr: float = 1e-3,
    seed: int = 42,
) -> dict:
    """
    Train a linear encoder on Binary data with InfoNCE.

    Evaluates every eval_every epochs; returns epoch snapshots at E0, E50, E100.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    binary_dataset = BinaryDataset(n_samples=n_samples, dim=data_dim, seed=seed)
    aug = BinaryAugmentation(flip_rate=0.001)
    train_wrapped = ContrastiveDatasetWrapper(binary_dataset, aug)

    train_loader = DataLoader(train_wrapped, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    plain_loader = DataLoader(binary_dataset, batch_size=256, shuffle=False, num_workers=0)

    model = LinearEncoder(in_dim=data_dim, out_dim=out_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    snapshots = {}

    # Evaluate at epoch 0 (before training)
    Z = extract_representations(model, plain_loader)
    snapshots['E0'] = gaussian_diagnostics(Z)

    for epoch in range(n_epochs):
        loss = train_one_epoch(model, train_loader, optimizer, temperature)
        if (epoch + 1) % eval_every == 0:
            print(f"  [Binary d={out_dim}] epoch {epoch+1}/{n_epochs} loss={loss:.4f}")
        # Capture snapshots at 50% and 100% of training
        half = n_epochs // 2
        if (epoch + 1) in [half, n_epochs]:
            Z = extract_representations(model, plain_loader)
            epoch_key = f'E{epoch+1}' if (epoch + 1) != n_epochs else f'E{n_epochs}'
            snapshots[epoch_key] = gaussian_diagnostics(Z)
        # Also capture at explicit epoch 50 and 100 for full runs
        if (epoch + 1) in [50, 100] and (epoch + 1) != half and (epoch + 1) != n_epochs:
            Z = extract_representations(model, plain_loader)
            snapshots[f'E{epoch+1}'] = gaussian_diagnostics(Z)

    return snapshots


# ---------------------------------------------------------------------------
# Figure 2: Alignment and uniformity vs. batch size / dimensionality
# ---------------------------------------------------------------------------

def plot_figure2(results: list[dict], save_path: str):
    """
    Reproduce Figure 2: alignment and uniformity vs. batch size / dimensionality.

    Left panel: Alignment vs. batch size for each dimension.
    Right panel: Uniformity vs. batch size for each dimension.
    """
    dims = sorted(set(r['out_dim'] for r in results))
    batch_sizes = sorted(set(r['batch_size'] for r in results))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Alignment vs. batch size ---
    ax = axes[0]
    for d in dims:
        subset = [r for r in results if r['out_dim'] == d]
        subset = sorted(subset, key=lambda r: r['batch_size'])
        bs = [r['batch_size'] for r in subset]
        aln = [r['alignment'] for r in subset]
        ax.plot(bs, aln, marker='o', label=f'd={d}')
    ax.set_xlabel('Batch size N')
    ax.set_ylabel('Alignment (mean positive cosine sim.)')
    ax.set_title('Alignment vs. Batch Size')
    ax.legend()
    ax.set_xscale('log', base=2)
    ax.grid(True, alpha=0.3)

    # --- Uniformity vs. batch size ---
    ax = axes[1]
    for d in dims:
        subset = [r for r in results if r['out_dim'] == d]
        subset = sorted(subset, key=lambda r: r['batch_size'])
        bs = [r['batch_size'] for r in subset]
        uni = [r['uniformity'] for r in subset]
        ax.plot(bs, uni, marker='o', label=f'd={d}')
    ax.set_xlabel('Batch size N')
    ax.set_ylabel('Uniformity (mean negative cosine sim.)')
    ax.set_title('Uniformity vs. Batch Size')
    ax.legend()
    ax.set_xscale('log', base=2)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Figure 2: Alignment and Uniformity (Laplace, Linear Encoder)', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved Figure 2 to {save_path}")


# ---------------------------------------------------------------------------
# Figure 3: Norm stats (CV vs. batch size for different dims)
# ---------------------------------------------------------------------------

def plot_figure3(results: list[dict], save_path: str):
    """
    Reproduce Figure 3: norm concentration (CV) vs. batch size / dimension.
    """
    dims = sorted(set(r['out_dim'] for r in results))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: CV vs. batch size, curves per dimension
    ax = axes[0]
    for d in dims:
        subset = [r for r in results if r['out_dim'] == d]
        subset = sorted(subset, key=lambda r: r['batch_size'])
        bs = [r['batch_size'] for r in subset]
        cvs = [r['cv'] for r in subset]
        ax.plot(bs, cvs, marker='o', label=f'd={d}')
    ax.set_xlabel('Batch size N')
    ax.set_ylabel('CV of norms')
    ax.set_title('Norm concentration (CV) vs. Batch Size')
    ax.legend()
    ax.set_xscale('log', base=2)
    ax.grid(True, alpha=0.3)

    # Right: AD statistic vs. batch size, curves per dimension
    ax = axes[1]
    for d in dims:
        subset = [r for r in results if r['out_dim'] == d]
        subset = sorted(subset, key=lambda r: r['batch_size'])
        bs = [r['batch_size'] for r in subset]
        ad = [r['ad_avg'] for r in subset]
        ax.plot(bs, ad, marker='o', label=f'd={d}')
    ax.axhline(0.752, color='r', linestyle='--', label='AD threshold (0.752)')
    ax.set_xlabel('Batch size N')
    ax.set_ylabel('Mean AD statistic')
    ax.set_title('Anderson-Darling statistic vs. Batch Size')
    ax.legend()
    ax.set_xscale('log', base=2)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Figure 3: Norm Concentration (Laplace, Linear Encoder)', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved Figure 3 to {save_path}")


# ---------------------------------------------------------------------------
# Table 1 printing
# ---------------------------------------------------------------------------

def print_table1_synthetic(laplace_diag: dict, gmm_diag: dict, binary_diag: dict):
    """Print Table 1 (synthetic data rows)."""
    header = (
        f"{'Metric':<25} | {'Laplace':>10} | {'GMM':>10} | "
        f"{'Binary E0':>10} | {'E50':>10} | {'E100':>10}"
    )
    sep = '-' * len(header)
    print('\n' + sep)
    print('Table 1: Gaussianity diagnostics (Synthetic)')
    print(sep)
    print(header)
    print(sep)

    b0 = binary_diag.get('E0', {})
    # In quick runs, E50/E100 may not exist; use last available snapshot
    all_keys = sorted([k for k in binary_diag.keys() if k != 'E0'],
                      key=lambda k: int(k[1:]))
    b50_key = 'E50' if 'E50' in binary_diag else (all_keys[len(all_keys)//2] if all_keys else 'E0')
    b100_key = 'E100' if 'E100' in binary_diag else (all_keys[-1] if all_keys else 'E0')
    b50 = binary_diag.get(b50_key, {})
    b100 = binary_diag.get(b100_key, {})

    rows = [
        ('CV', 'cv'),
        ('AD Avg. (<0.752)', 'ad_avg'),
        ('AD Norm. Feat. (%)', 'ad_norm_frac'),
        ('DP Avg. (>0.05)', 'dp_avg'),
        ('DP Norm. Feat. (%)', 'dp_norm_frac'),
    ]

    for label, key in rows:
        def fmt(d):
            v = d.get(key, float('nan'))
            if 'frac' in key:
                return f'{v*100:.1f}%'
            return f'{v:.3f}'
        print(
            f"{label:<25} | {fmt(laplace_diag):>10} | {fmt(gmm_diag):>10} | "
            f"{fmt(b0):>10} | {fmt(b50):>10} | {fmt(b100):>10}"
        )

    print(sep)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                        help='Quick run with fewer epochs and smaller grid')
    args = parser.parse_args()

    if args.quick:
        dims = [64, 128]
        batch_sizes = [32, 64, 128]
        n_epochs_laplace = 30
        n_epochs_gmm = 50  # GMM needs more epochs to converge
        n_epochs_binary = 100  # Need 100 to capture E0/E50/E100 snapshots
    else:
        dims = [32, 64, 128, 256]
        batch_sizes = [8, 16, 32, 48, 64, 96, 128]
        n_epochs_laplace = 150
        n_epochs_gmm = 100
        n_epochs_binary = 100

    # ---- Experiment 1: Laplace grid ----
    print('\n' + '='*60)
    print('EXPERIMENT 1: Laplace data, Linear Encoder')
    print('='*60)
    laplace_results = run_laplace_grid(
        dims=dims, batch_sizes=batch_sizes, n_epochs=n_epochs_laplace
    )
    # Save raw results
    with open(os.path.join(RESULTS_DIR, 'laplace_grid_results.json'), 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable = []
        for r in laplace_results:
            r2 = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in r.items()}
            serializable.append(r2)
        json.dump(serializable, f, indent=2)

    # For Table 1: pick d=256, largest batch size
    best_laplace = [r for r in laplace_results
                    if r['out_dim'] == max(dims) and r['batch_size'] == max(batch_sizes)][0]

    # Figures 2 and 3
    plot_figure2(laplace_results, os.path.join(RESULTS_DIR, 'fig2_alignment_uniformity.png'))
    plot_figure3(laplace_results, os.path.join(RESULTS_DIR, 'fig3_norm_stats.png'))

    # ---- Experiment 2: GMM ----
    print('\n' + '='*60)
    print('EXPERIMENT 2: Gaussian Mixture data, Linear Encoder')
    print('='*60)
    gmm_diag = run_gmm_experiment(n_epochs=n_epochs_gmm)
    print(f"GMM: CV={gmm_diag['cv']:.3f}, AD_avg={gmm_diag['ad_avg']:.3f}, "
          f"AD_norm%={gmm_diag['ad_norm_frac']*100:.1f}%, "
          f"DP_avg={gmm_diag['dp_avg']:.3f}, DP_norm%={gmm_diag['dp_norm_frac']*100:.1f}%")

    # ---- Experiment 3: Binary ----
    print('\n' + '='*60)
    print('EXPERIMENT 3: Binary data, Linear Encoder')
    print('='*60)
    binary_snapshots = run_binary_experiment(n_epochs=n_epochs_binary)
    for epoch_key, snap in binary_snapshots.items():
        print(f"  Binary {epoch_key}: CV={snap['cv']:.3f}, "
              f"AD_avg={snap['ad_avg']:.3f}, "
              f"AD_norm%={snap['ad_norm_frac']*100:.1f}%, "
              f"DP_avg={snap['dp_avg']:.3f}, "
              f"DP_norm%={snap['dp_norm_frac']*100:.1f}%")

    # ---- Print Table 1 ----
    print_table1_synthetic(best_laplace, gmm_diag, binary_snapshots)

    # Save summary
    with open(os.path.join(RESULTS_DIR, 'synthetic_table1.txt'), 'w') as f:
        import sys
        old_stdout = sys.stdout

        class Tee:
            def __init__(self, *files):
                self.files = files
            def write(self, s):
                for fi in self.files:
                    fi.write(s)
            def flush(self):
                for fi in self.files:
                    fi.flush()

        sys.stdout = Tee(sys.stdout, f)
        print_table1_synthetic(best_laplace, gmm_diag, binary_snapshots)
        sys.stdout = old_stdout

    print('\nAll synthetic experiments done. Results saved to', RESULTS_DIR)


if __name__ == '__main__':
    main()
