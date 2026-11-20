"""CIFAR-10 ResNet-18 Contrastive (SimCLR) vs. Supervised (Cross-Entropy) Experiment (Table 1).

Reproduces Section 5 'Contrastive vs. supervised training' & Appendix E.1:
- Dataset: CIFAR-10 (50k train, 10k test) with SimCLR-style augmentations
- Architecture: Two ResNet-18 encoders sharing IDENTICAL initialization and capacity
  (pretrained on ImageNet by default, per Appendix E.1), differing only in objective:
  * Contrastive: ResNet-18 backbone + SimCLR 2-layer MLP projection head, InfoNCE/NT-Xent (tau=0.1)
  * Supervised: ResNet-18 backbone + linear 10-class head, CrossEntropyLoss
- Optimization: Adam, lr = 1e-3, weight_decay = 1e-4, batch_size = 256, epochs = 100
- Evaluation: Gaussianity diagnostics (CV, AD, DP) on 512-dim backbone representations
  (both unnormalized & normalized), plus supervised top-1 accuracy and contrastive linear probe.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common.datasets import build_cifar10_loaders
from common.losses import InfoNCELoss, SimCLRNTXentLoss
from common.metrics import compute_norm_stats, evaluate_embeddings_both
from common.models import ResNet18SimCLR, ResNet18Supervised, build_matched_resnet18_pair
from common.utils import default_num_workers, ensure_dir, get_device, save_json, seed_everything


@torch.no_grad()
def extract_backbone_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract 512-dim penultimate backbone features and labels."""
    model.eval()
    feats: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        h = model.encode(images)
        feats.append(h.cpu().numpy())
        labels_list.append(labels.numpy())
    return np.concatenate(feats, axis=0), np.concatenate(labels_list, axis=0)


@torch.no_grad()
def evaluate_supervised_accuracy(
    model: ResNet18Supervised,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute top-1 classification accuracy on CIFAR-10 test set."""
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        _h, logits = model(images)
        preds = logits.argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += int(labels.size(0))
    return 100.0 * correct / max(total, 1)


def train_linear_probe(
    train_feats: np.ndarray,
    train_labels: np.ndarray,
    test_feats: np.ndarray,
    test_labels: np.ndarray,
    num_classes: int = 10,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-2,
    device: torch.device | None = None,
) -> float:
    """Train a standard linear probe on frozen contrastive representations to verify quality."""
    device = device or get_device()
    feat_dim = train_feats.shape[1]
    probe = nn.Linear(feat_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(
        torch.from_numpy(train_feats).float(),
        torch.from_numpy(train_labels).long(),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    probe.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            loss = criterion(probe(xb), yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    probe.eval()
    with torch.no_grad():
        logits = probe(torch.from_numpy(test_feats).float().to(device))
        preds = logits.argmax(dim=1).cpu().numpy()
    return float(100.0 * np.mean(preds == test_labels))


def run_cifar10_resnet18_experiment(
    data_dir: str | Path,
    output_dir: str | Path,
    pretrained: bool = True,
    cifar_stem: bool = False,
    contrastive_loss_type: str = "nt_xent",
    batch_size: int = 256,
    epochs: int = 100,
    eval_every: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    temperature: float = 0.1,
    num_workers: int | None = None,
    seed: int = 42,
    device_str: str = "cuda",
    download: bool = True,
) -> dict[str, Any]:
    """Train matched ResNet-18 models (Contrastive SimCLR vs. Supervised CE) on CIFAR-10."""
    seed_everything(seed)
    device = get_device(device_str)
    out = ensure_dir(output_dir)
    workers = default_num_workers() if num_workers is None else num_workers

    print(
        f"[CIFAR-10 ResNet-18] Device={device}, pretrained={pretrained}, cifar_stem={cifar_stem}, "
        f"loss={contrastive_loss_type}, bs={batch_size}, epochs={epochs}, lr={lr}, wd={weight_decay}, tau={temperature}"
    )

    loaders = build_cifar10_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=workers,
        use_imagenet_norm=pretrained,
        download=download,
    )

    contrastive_model, supervised_model = build_matched_resnet18_pair(
        pretrained=pretrained,
        cifar_stem=cifar_stem,
        num_classes=10,
        proj_hidden_dim=512,
        proj_output_dim=128,
    )
    contrastive_model = contrastive_model.to(device)
    supervised_model = supervised_model.to(device)

    # Verify identical initial backbone weights
    for (n1, p1), (n2, p2) in zip(
        contrastive_model.backbone.named_parameters(),
        supervised_model.backbone.named_parameters(),
    ):
        assert n1 == n2 and torch.equal(p1.data, p2.data), f"Backbone initialization mismatch at {n1}"

    # Evaluate shared initial representations (epoch 0)
    init_feats, _ = extract_backbone_embeddings(contrastive_model, loaders["test"], device)
    init_diag = evaluate_embeddings_both(init_feats)
    print(
        f"  Shared Init (E0) | CV={init_diag['unnormalized']['cv']:.4f} | "
        f"AD={init_diag['unnormalized']['ad_avg']:.3f} ({init_diag['unnormalized']['ad_norm_feat_pct']:.1f}%) | "
        f"DP={init_diag['unnormalized']['dp_avg']:.3f} ({init_diag['unnormalized']['dp_norm_feat_pct']:.1f}%)"
    )

    # ------------------------------------------------------------------
    # 1. Contrastive (SimCLR / InfoNCE) Training Loop
    # ------------------------------------------------------------------
    print("\n--- Training Contrastive ResNet-18 (SimCLR + InfoNCE) ---")
    if contrastive_loss_type == "nt_xent":
        contrastive_criterion: nn.Module = SimCLRNTXentLoss(temperature=temperature)
    elif contrastive_loss_type == "infonce":
        contrastive_criterion = InfoNCELoss(temperature=temperature, symmetric=False)
    elif contrastive_loss_type == "symmetric_infonce":
        contrastive_criterion = InfoNCELoss(temperature=temperature, symmetric=True)
    else:
        raise ValueError(f"Unsupported contrastive_loss_type: {contrastive_loss_type}")

    opt_contrastive = torch.optim.Adam(
        contrastive_model.parameters(), lr=lr, weight_decay=weight_decay
    )
    contrastive_history: list[dict[str, Any]] = []
    contrastive_losses: list[float] = []

    for epoch in range(1, epochs + 1):
        contrastive_model.train()
        running_loss = 0.0
        n_steps = 0
        for (v1, v2), _labels in loaders["train_contrastive"]:
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)
            _h1, z1 = contrastive_model(v1)
            _h2, z2 = contrastive_model(v2)
            loss = contrastive_criterion(z1, z2)

            opt_contrastive.zero_grad(set_to_none=True)
            loss.backward()
            opt_contrastive.step()
            running_loss += float(loss.item())
            n_steps += 1

        avg_loss = running_loss / max(n_steps, 1)
        contrastive_losses.append(avg_loss)

        if epoch % eval_every == 0 or epoch == epochs:
            test_h, _ = extract_backbone_embeddings(contrastive_model, loaders["test"], device)
            diag = evaluate_embeddings_both(test_h)
            un = diag["unnormalized"]
            contrastive_history.append({"epoch": epoch, "train_loss": avg_loss, "diagnostics": diag})
            print(
                f"  [Contrastive] Epoch {epoch:3d}/{epochs} | loss={avg_loss:.4f} | CV={un['cv']:.4f} | "
                f"AD={un['ad_avg']:.3f} ({un['ad_norm_feat_pct']:.1f}%) | "
                f"DP={un['dp_avg']:.3f} ({un['dp_norm_feat_pct']:.1f}%)"
            )

    # Final contrastive evaluation (512-dim backbone + projection head + linear probe)
    c_train_h, c_train_y = extract_backbone_embeddings(contrastive_model, loaders["train_eval"], device)
    c_test_h, c_test_y = extract_backbone_embeddings(contrastive_model, loaders["test"], device)
    c_final_diag = evaluate_embeddings_both(c_test_h)
    c_norms = compute_norm_stats(c_test_h)["norms"].astype(np.float32)

    # Also compute projection-head diagnostics for completeness
    contrastive_model.eval()
    proj_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for images, _labels in loaders["test"]:
            _h, z_proj = contrastive_model(images.to(device, non_blocking=True))
            proj_chunks.append(z_proj.cpu().numpy())
    c_proj_diag = evaluate_embeddings_both(np.concatenate(proj_chunks, axis=0))

    c_linear_probe_acc = train_linear_probe(
        c_train_h, c_train_y, c_test_h, c_test_y, device=device
    )
    print(f"  [Contrastive] Linear Probe Test Accuracy: {c_linear_probe_acc:.2f}%")
    torch.save(contrastive_model.state_dict(), out / "resnet18_contrastive.pt")

    # ------------------------------------------------------------------
    # 2. Supervised (Cross-Entropy) Training Loop
    # ------------------------------------------------------------------
    print("\n--- Training Supervised ResNet-18 (Cross-Entropy) ---")
    sup_criterion = nn.CrossEntropyLoss()
    opt_supervised = torch.optim.Adam(
        supervised_model.parameters(), lr=lr, weight_decay=weight_decay
    )
    supervised_history: list[dict[str, Any]] = []
    supervised_losses: list[float] = []

    for epoch in range(1, epochs + 1):
        supervised_model.train()
        running_loss = 0.0
        n_steps = 0
        for images, labels in loaders["train_supervised"]:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            _h, logits = supervised_model(images)
            loss = sup_criterion(logits, labels)

            opt_supervised.zero_grad(set_to_none=True)
            loss.backward()
            opt_supervised.step()
            running_loss += float(loss.item())
            n_steps += 1

        avg_loss = running_loss / max(n_steps, 1)
        supervised_losses.append(avg_loss)

        if epoch % eval_every == 0 or epoch == epochs:
            test_h, _ = extract_backbone_embeddings(supervised_model, loaders["test"], device)
            diag = evaluate_embeddings_both(test_h)
            test_acc = evaluate_supervised_accuracy(supervised_model, loaders["test"], device)
            un = diag["unnormalized"]
            supervised_history.append(
                {
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "test_accuracy": test_acc,
                    "diagnostics": diag,
                }
            )
            print(
                f"  [Supervised]  Epoch {epoch:3d}/{epochs} | loss={avg_loss:.4f} | acc={test_acc:.2f}% | "
                f"CV={un['cv']:.4f} | AD={un['ad_avg']:.3f} ({un['ad_norm_feat_pct']:.1f}%) | "
                f"DP={un['dp_avg']:.4f} ({un['dp_norm_feat_pct']:.1f}%)"
            )

    s_test_h, _ = extract_backbone_embeddings(supervised_model, loaders["test"], device)
    s_final_diag = evaluate_embeddings_both(s_test_h)
    s_final_acc = evaluate_supervised_accuracy(supervised_model, loaders["test"], device)
    s_norms = compute_norm_stats(s_test_h)["norms"].astype(np.float32)
    torch.save(supervised_model.state_dict(), out / "resnet18_supervised.pt")

    summary = {
        "config": {
            "pretrained": pretrained,
            "cifar_stem": cifar_stem,
            "contrastive_loss_type": contrastive_loss_type,
            "batch_size": batch_size,
            "epochs": epochs,
            "eval_every": eval_every,
            "lr": lr,
            "weight_decay": weight_decay,
            "temperature": temperature,
            "seed": seed,
        },
        "shared_init": init_diag,
        "contrastive": {
            "loss_history": contrastive_losses,
            "eval_history": contrastive_history,
            "final": c_final_diag,
            "final_projector": c_proj_diag,
            "linear_probe_test_accuracy": c_linear_probe_acc,
        },
        "supervised": {
            "loss_history": supervised_losses,
            "eval_history": supervised_history,
            "final": s_final_diag,
            "final_test_accuracy": s_final_acc,
        },
    }
    save_json(summary, out / "cifar10_resnet18_results.json")
    np.savez_compressed(
        out / "cifar10_resnet18_norms.npz",
        contrastive_norms=c_norms,
        supervised_norms=s_norms,
    )
    print(f"[CIFAR-10 ResNet-18] Saved results and checkpoints to {out}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CIFAR-10 ResNet-18 Contrastive vs. Supervised experiment.")
    parser.add_argument("--data-dir", type=str, default=str(ROOT_DIR / "data" / "cifar10"))
    parser.add_argument("--output-dir", type=str, default=str(ROOT_DIR / "results" / "cifar10_resnet18"))
    parser.add_argument("--no-pretrained", action="store_true", help="Initialize ResNet-18 randomly instead of ImageNet weights.")
    parser.add_argument("--cifar-stem", action="store_true", help="Use 3x3 stride-1 conv1 and drop maxpool.")
    parser.add_argument(
        "--contrastive-loss",
        type=str,
        choices=["nt_xent", "infonce", "symmetric_infonce"],
        default="nt_xent",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_cifar10_resnet18_experiment(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        pretrained=not args.no_pretrained,
        cifar_stem=args.cifar_stem,
        contrastive_loss_type=args.contrastive_loss,
        batch_size=args.batch_size,
        epochs=args.epochs,
        eval_every=args.eval_every,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        num_workers=args.num_workers,
        seed=args.seed,
        device_str=args.device,
        download=not args.no_download,
    )

