"""
CIFAR-10 experiments — ResNet-18, Contrastive vs. Supervised (Table 1).

Trains two ResNet-18 models on CIFAR-10:
  1. Contrastive: InfoNCE objective, SimCLR-style augmentations
  2. Supervised: Cross-entropy objective, standard augmentations

Evaluates Gaussian diagnostics (CV, AD, DP) on representations extracted
from the test set to replicate Table 1 (ResNet-18 columns).

Paper spec (Appendix E.1):
  - Architecture: ResNet-18 (He et al., 2016), SimCLR protocol for contrastive
  - Contrastive: InfoNCE, τ=0.1
  - Supervised: Cross-entropy
  - Optimizer (both): Adam, lr=1e-3, weight_decay=1e-4
  - Batch size: 256
  - Epochs: 100

Outputs:
  - results/cifar10_resnet_table1.txt           — Table 1 (ResNet-18 rows)
  - results/cifar10_resnet_contrastive_diag.json
  - results/cifar10_resnet_supervised_diag.json
"""

from __future__ import annotations

import os
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from augmentations import SimCLRAugmentation, StandardCIFAR10Transform, CIFAR10ContrastiveWrapper
from models import ResNet18Encoder, ResNet18Supervised
from infonce_loss import infonce_loss
from utils import gaussian_diagnostics


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def get_contrastive_loader(batch_size: int = 256, train: bool = True) -> DataLoader:
    """SimCLR-style paired augmentation loader for CIFAR-10."""
    aug = SimCLRAugmentation(size=32)
    base = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=train, download=True, transform=None
    )
    ds = CIFAR10ContrastiveWrapper(base, aug)
    return DataLoader(ds, batch_size=batch_size, shuffle=train,
                      num_workers=2, pin_memory=True, drop_last=True)


def get_supervised_train_loader(batch_size: int = 256) -> DataLoader:
    """Standard training loader for supervised CIFAR-10."""
    transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010]),
    ])
    ds = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True,
                                       download=True, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=2, pin_memory=True, drop_last=True)


def get_eval_loader(batch_size: int = 256) -> DataLoader:
    """Plain test loader for evaluation."""
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010]),
    ])
    ds = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False,
                                       download=True, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=2, pin_memory=True)


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def train_one_epoch_contrastive(
    model: ResNet18Encoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    temperature: float,
) -> float:
    """One epoch of InfoNCE training."""
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


def train_one_epoch_supervised(
    model: ResNet18Supervised,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
) -> float:
    """One epoch of cross-entropy training."""
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def supervised_accuracy(model: ResNet18Supervised, loader: DataLoader) -> float:
    """Compute classification accuracy of supervised model."""
    model.eval()
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


# ---------------------------------------------------------------------------
# Representation extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_contrastive_reprs(
    model: ResNet18Encoder, loader: DataLoader
) -> torch.Tensor:
    """
    Extract representations from contrastive ResNet-18.
    Uses the projection head output (unnormalized z).
    """
    model.eval()
    all_z = []
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        z = model(imgs)
        all_z.append(z.cpu())
    return torch.cat(all_z, dim=0)


@torch.no_grad()
def extract_supervised_reprs(
    model: ResNet18Supervised, loader: DataLoader
) -> torch.Tensor:
    """
    Extract representations from supervised ResNet-18.
    Uses pre-classification backbone features.
    """
    model.eval()
    all_z = []
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        z = model.get_representations(imgs)
        all_z.append(z.cpu())
    return torch.cat(all_z, dim=0)


# ---------------------------------------------------------------------------
# Full training runs
# ---------------------------------------------------------------------------

def run_contrastive_resnet18(
    out_dim: int = 256,
    batch_size: int = 256,
    n_epochs: int = 100,
    temperature: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
) -> dict:
    """Train ResNet-18 with InfoNCE and return Gaussian diagnostics."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"\n--- Contrastive ResNet-18 (InfoNCE, τ={temperature}) ---")

    train_loader = get_contrastive_loader(batch_size=batch_size, train=True)
    eval_loader = get_eval_loader(batch_size=batch_size)

    model = ResNet18Encoder(out_dim=out_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(n_epochs):
        loss = train_one_epoch_contrastive(model, train_loader, optimizer, temperature)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}, loss={loss:.4f}")

    # Evaluate
    Z = extract_contrastive_reprs(model, eval_loader)
    diag = gaussian_diagnostics(Z)

    print(f"  [Contrastive] CV={diag['cv']:.3f}, "
          f"AD_avg={diag['ad_avg']:.3f}, AD%={diag['ad_norm_frac']*100:.1f}%, "
          f"DP_avg={diag['dp_avg']:.3f}, DP%={diag['dp_norm_frac']*100:.1f}%")

    # Save checkpoint
    torch.save(model.state_dict(),
               os.path.join(RESULTS_DIR, 'resnet18_contrastive.pt'))
    return diag


def run_supervised_resnet18(
    num_classes: int = 10,
    batch_size: int = 256,
    n_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
) -> dict:
    """Train ResNet-18 with cross-entropy and return Gaussian diagnostics."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"\n--- Supervised ResNet-18 (cross-entropy) ---")

    train_loader = get_supervised_train_loader(batch_size=batch_size)
    eval_loader = get_eval_loader(batch_size=batch_size)

    model = ResNet18Supervised(num_classes=num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(n_epochs):
        loss = train_one_epoch_supervised(model, train_loader, optimizer)
        if (epoch + 1) % 10 == 0:
            acc = supervised_accuracy(model, eval_loader)
            print(f"  Epoch {epoch+1}/{n_epochs}, loss={loss:.4f}, test_acc={acc:.3f}")

    # Evaluate Gaussian diagnostics on backbone representations
    Z = extract_supervised_reprs(model, eval_loader)
    diag = gaussian_diagnostics(Z)

    print(f"  [Supervised]  CV={diag['cv']:.3f}, "
          f"AD_avg={diag['ad_avg']:.3f}, AD%={diag['ad_norm_frac']*100:.1f}%, "
          f"DP_avg={diag['dp_avg']:.3f}, DP%={diag['dp_norm_frac']*100:.1f}%")

    # Save checkpoint
    torch.save(model.state_dict(),
               os.path.join(RESULTS_DIR, 'resnet18_supervised.pt'))
    return diag


# ---------------------------------------------------------------------------
# Table 1 output
# ---------------------------------------------------------------------------

def print_table1_cifar(contrastive_diag: dict, supervised_diag: dict):
    """Print Table 1 (CIFAR-10 ResNet-18 rows)."""
    header = (
        f"{'Metric':<25} | {'Supervised (ResNet-18)':>22} | "
        f"{'Contrastive (ResNet-18)':>24}"
    )
    sep = '-' * len(header)
    print('\n' + sep)
    print('Table 1: Gaussianity diagnostics (CIFAR-10, ResNet-18)')
    print(sep)
    print(header)
    print(sep)

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
        print(f"{label:<25} | {fmt(supervised_diag):>22} | {fmt(contrastive_diag):>24}")

    # Gaussian? based on paper thresholds
    sup_gaussian = (
        supervised_diag.get('ad_avg', 999) < 0.752 and
        supervised_diag.get('dp_avg', 0) > 0.05
    )
    con_gaussian = (
        contrastive_diag.get('ad_avg', 999) < 0.752 and
        contrastive_diag.get('dp_avg', 0) > 0.05
    )
    print(f"{'Gaussian?':<25} | {'✓' if sup_gaussian else '✗':>22} | "
          f"{'✓' if con_gaussian else '✗':>24}")
    print(sep)

    # Paper reference values (Table 1):
    print('\n  Paper reference values (Table 1):')
    print('  Supervised : CV=0.5,  AD_avg=3.3,  AD%=6.2%,  DP_avg=0.041, DP%=3.9%')
    print('  Contrastive: CV=0.09, AD_avg=0.43, AD%=96.1%, DP_avg=0.39,  DP%=94.5%')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                        help='Quick run with fewer epochs')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--skip-supervised', action='store_true',
                        help='Skip supervised training (run contrastive only)')
    args = parser.parse_args()

    n_epochs = 5 if args.quick else args.epochs

    print('\n' + '='*60)
    print('EXPERIMENT 5: CIFAR-10, ResNet-18, Contrastive vs. Supervised')
    print('='*60)
    print(f"Device: {DEVICE}")
    print(f"Epochs: {n_epochs}, Batch size: {args.batch_size}")

    # Run contrastive
    contrastive_diag = run_contrastive_resnet18(n_epochs=n_epochs, batch_size=args.batch_size)

    # Run supervised
    if not args.skip_supervised:
        supervised_diag = run_supervised_resnet18(n_epochs=n_epochs, batch_size=args.batch_size)
    else:
        supervised_diag = {}
        print('Supervised training skipped.')

    # Save diagnostics
    def to_serializable(d):
        return {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in d.items()}

    with open(os.path.join(RESULTS_DIR, 'cifar10_resnet_contrastive_diag.json'), 'w') as f:
        json.dump(to_serializable(contrastive_diag), f, indent=2)

    if supervised_diag:
        with open(os.path.join(RESULTS_DIR, 'cifar10_resnet_supervised_diag.json'), 'w') as f:
            json.dump(to_serializable(supervised_diag), f, indent=2)

    # Print Table 1
    print_table1_cifar(contrastive_diag, supervised_diag if supervised_diag else {
        'cv': float('nan'), 'ad_avg': float('nan'), 'ad_norm_frac': float('nan'),
        'dp_avg': float('nan'), 'dp_norm_frac': float('nan'),
    })

    # Save table to file
    with open(os.path.join(RESULTS_DIR, 'cifar10_resnet_table1.txt'), 'w') as f:
        import sys

        class Tee:
            def __init__(self, *files):
                self.files = files
            def write(self, s):
                for fi in self.files:
                    fi.write(s)
            def flush(self):
                for fi in self.files:
                    fi.flush()

        old = sys.stdout
        sys.stdout = Tee(sys.stdout, f)
        print_table1_cifar(
            contrastive_diag,
            supervised_diag if supervised_diag else {
                'cv': float('nan'), 'ad_avg': float('nan'), 'ad_norm_frac': float('nan'),
                'dp_avg': float('nan'), 'dp_norm_frac': float('nan'),
            }
        )
        sys.stdout = old

    print('\nCIFAR-10 ResNet-18 experiments done. Results in', RESULTS_DIR)


if __name__ == '__main__':
    main()
