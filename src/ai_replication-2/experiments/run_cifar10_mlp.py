"""CIFAR-10 Two-Layer MLP InfoNCE Training Dynamics Experiment (Figure 4).

Reproduces Section 5 'CIFAR-10 experiments' & Appendix E.1:
- Dataset: CIFAR-10 (50k train, 10k test)
- Augmentations: SimCLR-style pipeline (RandomResizedCrop 32x32 scale=(0.2, 1.0),
  RandomHorizontalFlip, ColorJitter(0.8, 0.8, 0.8, 0.2), RandomGrayscale(p=0.2))
- Architecture: Two-layer MLP with a single ReLU nonlinear activation
- Objective: InfoNCE with temperature tau = 0.1
- Optimizer: Adam, lr = 1e-3, weight_decay = 1e-4, batch_size = 256, epochs = 100
- Evaluation: Test-set CV, AD statistic, and DP passing feature fraction tracked across epochs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common.datasets import build_cifar10_loaders
from common.losses import InfoNCELoss
from common.metrics import compute_norm_stats, evaluate_embeddings_both
from common.models import TwoLayerMLPEncoder
from common.utils import default_num_workers, ensure_dir, get_device, save_json, seed_everything


@torch.no_grad()
def extract_mlp_embeddings(
    model: TwoLayerMLPEncoder,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Extract unnormalized test-set representations from the two-layer MLP."""
    model.eval()
    chunks: list[np.ndarray] = []
    for images, _labels in loader:
        images = images.to(device, non_blocking=True)
        z = model(images)
        chunks.append(z.cpu().numpy())
    return np.concatenate(chunks, axis=0)


def run_cifar10_mlp_experiment(
    data_dir: str | Path,
    output_dir: str | Path,
    hidden_dim: int = 512,
    repr_dim: int = 256,
    batch_size: int = 256,
    epochs: int = 100,
    eval_every: int = 1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    temperature: float = 0.1,
    num_workers: int | None = None,
    seed: int = 42,
    device_str: str = "cuda",
    download: bool = True,
) -> dict[str, Any]:
    """Train a 2-layer ReLU MLP with InfoNCE on CIFAR-10 and track Gaussianity over epochs."""
    seed_everything(seed)
    device = get_device(device_str)
    out = ensure_dir(output_dir)
    workers = default_num_workers() if num_workers is None else num_workers

    print(
        f"[CIFAR-10 MLP] Device={device}, hidden_dim={hidden_dim}, repr_dim={repr_dim}, "
        f"bs={batch_size}, epochs={epochs}, lr={lr}, wd={weight_decay}, tau={temperature}"
    )

    loaders = build_cifar10_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=workers,
        use_imagenet_norm=False,
        download=download,
    )
    train_loader = loaders["train_contrastive"]
    test_loader = loaders["test"]

    model = TwoLayerMLPEncoder(
        input_dim=3 * 32 * 32,
        hidden_dim=hidden_dim,
        output_dim=repr_dim,
    ).to(device)
    criterion = InfoNCELoss(temperature=temperature, symmetric=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: list[dict[str, Any]] = []
    saved_norms: dict[str, np.ndarray] = {}

    def _evaluate_and_log(ep: int, train_loss: float | None) -> None:
        z_test = extract_mlp_embeddings(model, test_loader, device)
        diag = evaluate_embeddings_both(z_test)
        un = diag["unnormalized"]
        entry = {
            "epoch": ep,
            "train_loss": train_loss,
            "diagnostics": diag,
        }
        history.append(entry)
        if ep in {0, 10, 25, 50, 75, epochs}:
            saved_norms[f"epoch_{ep}_norms"] = compute_norm_stats(z_test)["norms"].astype(np.float32)
        loss_str = f"{train_loss:.4f}" if train_loss is not None else "  N/A "
        print(
            f"  Epoch {ep:3d}/{epochs} | loss={loss_str} | CV={un['cv']:.4f} | "
            f"AD={un['ad_avg']:.3f} ({un['ad_norm_feat_pct']:.1f}%) | "
            f"DP={un['dp_avg']:.3f} ({un['dp_norm_feat_pct']:.1f}%)"
        )

    _evaluate_and_log(0, None)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_steps = 0
        for (v1, v2), _labels in train_loader:
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)
            z1 = model(v1)
            z2 = model(v2)
            loss = criterion(z1, z2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_steps += 1

        avg_loss = total_loss / max(n_steps, 1)
        if epoch % eval_every == 0 or epoch == epochs:
            _evaluate_and_log(epoch, avg_loss)

    torch.save(model.state_dict(), out / "cifar10_mlp_encoder.pt")

    summary = {
        "config": {
            "hidden_dim": hidden_dim,
            "repr_dim": repr_dim,
            "batch_size": batch_size,
            "epochs": epochs,
            "eval_every": eval_every,
            "lr": lr,
            "weight_decay": weight_decay,
            "temperature": temperature,
            "seed": seed,
        },
        "history": history,
        "final": history[-1],
    }
    save_json(summary, out / "cifar10_mlp_results.json")
    np.savez_compressed(out / "cifar10_mlp_norms.npz", **saved_norms)
    print(f"[CIFAR-10 MLP] Saved results and checkpoint to {out}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CIFAR-10 2-layer MLP InfoNCE dynamics experiment.")
    parser.add_argument("--data-dir", type=str, default=str(ROOT_DIR / "data" / "cifar10"))
    parser.add_argument("--output-dir", type=str, default=str(ROOT_DIR / "results" / "cifar10_mlp"))
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--repr-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=1)
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
    run_cifar10_mlp_experiment(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        hidden_dim=args.hidden_dim,
        repr_dim=args.repr_dim,
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

