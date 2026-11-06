"""
CIFAR-10 experiments — Two-layer MLP with InfoNCE (Figure 4).

Tracks Gaussianity metrics over training epochs to show the emergence of
Gaussian structure as training progresses.

Paper spec (Appendix E.1):
  - Two-layer MLP with single ReLU: Linear(3072, 512) → ReLU → Linear(512, 256)
  - InfoNCE, τ=0.1
  - Adam, lr=1e-3, weight_decay=1e-4
  - Batch size: 256
  - Epochs: 100
  - SimCLR augmentations (Appendix E.1)
  - Evaluate on test set throughout training

Outputs:
  - results/fig4_cifar10_mlp_dynamics.png  — Figure 4
  - results/cifar10_mlp_dynamics.json      — raw metric traces
"""

from __future__ import annotations

import os
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from augmentations import SimCLRAugmentation, StandardCIFAR10Transform, CIFAR10ContrastiveWrapper
from models import MLP
from infonce_loss import infonce_loss
from utils import (
    gaussian_diagnostics,
    coefficient_of_variation,
)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_cifar10_contrastive_loader(batch_size: int = 256, train: bool = True):
    """Return a DataLoader that yields (view1, view2, label) pairs via SimCLR augmentations."""
    aug = SimCLRAugmentation(size=32)
    base_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=train, download=True,
        transform=None,  # Raw PIL images; augmentation applied in wrapper
    )
    contrastive_ds = CIFAR10ContrastiveWrapper(base_dataset, aug)
    return DataLoader(
        contrastive_ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )


def get_cifar10_eval_loader(batch_size: int = 256):
    """Return a DataLoader for the test set (plain transforms) for evaluation."""
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010]),
    ])
    test_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform
    )
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_representations_cifar(model: nn.Module, loader: DataLoader) -> torch.Tensor:
    """Extract unnormalized MLP representations from a CIFAR-10 eval loader."""
    model.eval()
    all_z = []
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        z = model(imgs)  # MLP flattens internally
        all_z.append(z.cpu())
    return torch.cat(all_z, dim=0)


def train_one_epoch_cifar(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    temperature: float,
) -> float:
    """Train for one epoch on paired CIFAR-10 views; return mean loss."""
    model.train()
    total_loss = 0.0
    for v1, v2, _ in loader:
        v1, v2 = v1.to(DEVICE), v2.to(DEVICE)
        z1 = model(v1)
        z2 = model(v2)
        loss = infonce_loss(z1, z2, temperature=temperature)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Main experiment: MLP dynamics
# ---------------------------------------------------------------------------

def run_cifar10_mlp(
    hidden_dim: int = 512,
    out_dim: int = 256,
    batch_size: int = 256,
    n_epochs: int = 100,
    temperature: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    eval_every: int = 5,
    seed: int = 42,
) -> dict:
    """
    Train a two-layer MLP on CIFAR-10 with InfoNCE and track Gaussianity over training.

    Returns a dict with per-epoch metric traces.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data
    train_loader = get_cifar10_contrastive_loader(batch_size=batch_size, train=True)
    eval_loader = get_cifar10_eval_loader(batch_size=batch_size)

    # Model: two-layer MLP, input = 3*32*32 = 3072
    model = MLP(in_dim=3 * 32 * 32, hidden_dim=hidden_dim, out_dim=out_dim).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    history = {
        'epoch': [],
        'loss': [],
        'cv': [],
        'ad_avg': [],
        'ad_norm_frac': [],
        'dp_avg': [],
        'dp_norm_frac': [],
        'mean_norm': [],
    }

    for epoch in range(n_epochs):
        loss = train_one_epoch_cifar(model, train_loader, optimizer, temperature)

        if (epoch + 1) % eval_every == 0 or epoch == 0:
            Z = extract_representations_cifar(model, eval_loader)
            diag = gaussian_diagnostics(Z)

            history['epoch'].append(epoch + 1)
            history['loss'].append(loss)
            history['cv'].append(diag['cv'])
            history['ad_avg'].append(diag['ad_avg'])
            history['ad_norm_frac'].append(diag['ad_norm_frac'])
            history['dp_avg'].append(diag['dp_avg'])
            history['dp_norm_frac'].append(diag['dp_norm_frac'])
            history['mean_norm'].append(diag['mean_norm'])

            print(
                f"  Epoch {epoch+1:3d}/{n_epochs} | loss={loss:.4f} | "
                f"CV={diag['cv']:.3f} | AD_avg={diag['ad_avg']:.3f} | "
                f"AD%={diag['ad_norm_frac']*100:.1f}% | "
                f"DP%={diag['dp_norm_frac']*100:.1f}%"
            )

    return history


# ---------------------------------------------------------------------------
# Figure 4: Training dynamics plot
# ---------------------------------------------------------------------------

def plot_figure4(history: dict, save_path: str):
    """
    Reproduce Figure 4: Gaussianity metrics over CIFAR-10 training epochs.

    Left: CV (norm concentration) over epochs
    Middle: AD statistic over epochs
    Right: Fraction of coordinates passing DP test over epochs
    """
    epochs = history['epoch']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Left: CV
    axes[0].plot(epochs, history['cv'], 'b-o', markersize=4)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('CV of norms')
    axes[0].set_title('Norm Concentration (CV)')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(bottom=0)

    # Middle: AD statistic
    axes[1].plot(epochs, history['ad_avg'], 'r-o', markersize=4)
    axes[1].axhline(0.752, color='r', linestyle='--', alpha=0.5, label='AD threshold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Mean AD statistic')
    axes[1].set_title('Anderson-Darling Statistic')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(bottom=0)

    # Right: DP fraction passing
    axes[2].plot(epochs, [v * 100 for v in history['dp_norm_frac']], 'g-o', markersize=4)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('% coordinates passing DP test (p>0.05)')
    axes[2].set_title("D'Agostino-Pearson: Normal Features (%)")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 105)

    plt.suptitle('Figure 4: CIFAR-10 MLP — Gaussianity Over Training', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved Figure 4 to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                        help='Quick run: fewer epochs')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()

    n_epochs = 10 if args.quick else args.epochs

    print('\n' + '='*60)
    print('EXPERIMENT 4: CIFAR-10, Two-Layer MLP, InfoNCE (Figure 4)')
    print('='*60)
    print(f"Device: {DEVICE}")

    history = run_cifar10_mlp(
        n_epochs=n_epochs,
        batch_size=args.batch_size,
        eval_every=max(1, n_epochs // 20),
    )

    # Save results
    with open(os.path.join(RESULTS_DIR, 'cifar10_mlp_dynamics.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # Figure 4
    plot_figure4(history, os.path.join(RESULTS_DIR, 'fig4_cifar10_mlp_dynamics.png'))

    print('\nFinal metrics:')
    if history['epoch']:
        print(f"  CV={history['cv'][-1]:.3f}")
        print(f"  AD_avg={history['ad_avg'][-1]:.3f}")
        print(f"  AD_norm%={history['ad_norm_frac'][-1]*100:.1f}%")
        print(f"  DP_avg={history['dp_avg'][-1]:.3f}")
        print(f"  DP_norm%={history['dp_norm_frac'][-1]*100:.1f}%")

    print('\nCIFAR-10 MLP experiment done. Results in', RESULTS_DIR)


if __name__ == '__main__':
    main()
