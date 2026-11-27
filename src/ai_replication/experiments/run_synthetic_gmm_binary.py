"""Additional Synthetic Data Experiments: GMM & Discrete Sparse Binary (Table 1).

Reproduces Appendix E.1 'Additional synthetic data experiments':
1. Gaussian Mixture Model (GMM):
   - 10k samples in R^{1024} from 25 equally weighted isotropic Gaussian components
     with fixed random means (plus held-out test set from the same fixed components).
   - Positive pairs: two views sampled independently from the same mixture component.
   - Linear encoder -> 256-dim representation, InfoNCE (tau=0.1), Adam (lr=1e-3), 100 epochs.
2. Discrete Sparse Binary:
   - 10k samples of sparse binary vectors in {0, 1}^{1024} (plus held-out test set).
   - Positive pairs: independently flip 0.1% of zero entries to ones.
   - Linear encoder -> 256-dim representation, InfoNCE (tau=0.1), Adam (lr=1e-3), 100 epochs.
   - Evaluated every 10 epochs (including E0, E50, E100 reported in Table 1).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common.datasets import (
    GMMSyntheticDataset,
    augment_sparse_binary_pair,
    generate_sparse_binary_dataset,
)
from common.losses import InfoNCELoss
from common.metrics import evaluate_embeddings_both
from common.models import LinearEncoder
from common.utils import ensure_dir, get_device, save_json, seed_everything


def run_gmm_experiment(
    n_train: int = 10_000,
    n_test: int = 5_000,
    d_data: int = 1024,
    n_components: int = 25,
    mean_std: float = 0.05,
    component_std: float = 1.0,
    repr_dim: int = 256,
    batch_size: int = 128,
    epochs: int = 100,
    lr: float = 1e-3,
    temperature: float = 0.1,
    seed: int = 42,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Train and evaluate a linear InfoNCE encoder on the 25-component GMM dataset."""
    seed_everything(seed)
    device = device or get_device()
    print(
        f"[Synthetic GMM] Device={device}, components={n_components}, d_data={d_data}, "
        f"repr_dim={repr_dim}, epochs={epochs}, bs={batch_size}"
    )

    gmm = GMMSyntheticDataset(
        n_train=n_train,
        n_test=n_test,
        d_data=d_data,
        n_components=n_components,
        mean_std=mean_std,
        component_std=component_std,
        seed=seed,
    )
    train_loader = gmm.get_train_loader(batch_size=batch_size)
    x_test_dev = gmm.x_test.to(device)

    model = LinearEncoder(input_dim=d_data, output_dim=repr_dim).to(device)
    criterion = InfoNCELoss(temperature=temperature, symmetric=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Evaluate at initialization (epoch 0)
    model.eval()
    with torch.no_grad():
        init_diag = evaluate_embeddings_both(model(x_test_dev).cpu().numpy())

    loss_history: list[float] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_steps = 0
        for (comp_ids,) in train_loader:
            x1, x2 = gmm.sample_positive_pair_batch(comp_ids, device=device)
            z1 = model(x1)
            z2 = model(x2)
            loss = criterion(z1, z2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_steps += 1
        loss_history.append(total_loss / max(n_steps, 1))

    model.eval()
    with torch.no_grad():
        final_z = model(x_test_dev).cpu().numpy()
    final_diag = evaluate_embeddings_both(final_z)
    un = final_diag["unnormalized"]
    print(
        f"  GMM Final (E{epochs}) | loss={loss_history[-1]:.4f} | CV={un['cv']:.4f} | "
        f"AD={un['ad_avg']:.3f} ({un['ad_norm_feat_pct']:.1f}%) | "
        f"DP={un['dp_avg']:.3f} ({un['dp_norm_feat_pct']:.1f}%)"
    )

    return {
        "config": {
            "n_train": n_train,
            "n_test": n_test,
            "d_data": d_data,
            "n_components": n_components,
            "mean_std": mean_std,
            "component_std": component_std,
            "repr_dim": repr_dim,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "temperature": temperature,
            "seed": seed,
        },
        "loss_history": loss_history,
        "epoch_0": init_diag,
        "final": final_diag,
    }


def run_sparse_binary_experiment(
    n_train: int = 10_000,
    n_test: int = 5_000,
    d_data: int = 1024,
    sparsity_p: float = 0.05,
    flip_zero_prob: float = 0.001,
    repr_dim: int = 256,
    batch_size: int = 128,
    epochs: int = 100,
    eval_every: int = 10,
    lr: float = 1e-3,
    temperature: float = 0.1,
    seed: int = 42,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Train and evaluate a linear InfoNCE encoder on discrete sparse binary vectors."""
    seed_everything(seed)
    device = device or get_device()
    print(
        f"[Synthetic Sparse Binary] Device={device}, d_data={d_data}, sparsity_p={sparsity_p}, "
        f"flip_prob={flip_zero_prob}, repr_dim={repr_dim}, epochs={epochs}, bs={batch_size}"
    )

    x_train, x_test = generate_sparse_binary_dataset(
        n_train=n_train,
        n_test=n_test,
        d_data=d_data,
        sparsity_p=sparsity_p,
        seed=seed,
    )
    x_train_dev = x_train.to(device)
    x_test_dev = x_test.to(device)

    model = LinearEncoder(input_dim=d_data, output_dim=repr_dim).to(device)
    criterion = InfoNCELoss(temperature=temperature, symmetric=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    eval_history: dict[str, Any] = {}
    loss_history: list[float] = []

    def _eval_at_epoch(ep: int) -> None:
        model.eval()
        with torch.no_grad():
            z = model(x_test_dev).cpu().numpy()
        diag = evaluate_embeddings_both(z)
        eval_history[f"epoch_{ep}"] = diag
        un = diag["unnormalized"]
        print(
            f"  Binary E{ep:3d} | CV={un['cv']:.4f} | "
            f"AD={un['ad_avg']:.3f} ({un['ad_norm_feat_pct']:.1f}%) | "
            f"DP={un['dp_avg']:.3f} ({un['dp_norm_feat_pct']:.1f}%)"
        )

    _eval_at_epoch(0)

    n_batches = n_train // batch_size
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            xb = x_train_dev[idx]
            v1, v2 = augment_sparse_binary_pair(xb, flip_zero_prob=flip_zero_prob)
            z1 = model(v1)
            z2 = model(v2)
            loss = criterion(z1, z2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        loss_history.append(epoch_loss / max(n_batches, 1))

        if epoch % eval_every == 0 or epoch == epochs:
            _eval_at_epoch(epoch)

    return {
        "config": {
            "n_train": n_train,
            "n_test": n_test,
            "d_data": d_data,
            "sparsity_p": sparsity_p,
            "flip_zero_prob": flip_zero_prob,
            "repr_dim": repr_dim,
            "batch_size": batch_size,
            "epochs": epochs,
            "eval_every": eval_every,
            "lr": lr,
            "temperature": temperature,
            "seed": seed,
        },
        "loss_history": loss_history,
        "epochs_eval": eval_history,
    }


def run_additional_synthetic_experiments(
    output_dir: str | Path,
    epochs: int = 100,
    batch_size: int = 128,
    repr_dim: int = 256,
    lr: float = 1e-3,
    temperature: float = 0.1,
    mean_std: float = 0.05,
    sparsity_p: float = 0.05,
    seed: int = 42,
    device_str: str = "cuda",
) -> dict[str, Any]:
    """Run both GMM and Sparse Binary synthetic experiments and save JSON outputs."""
    out = ensure_dir(output_dir)
    device = get_device(device_str)

    gmm_res = run_gmm_experiment(
        mean_std=mean_std,
        repr_dim=repr_dim,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        temperature=temperature,
        seed=seed,
        device=device,
    )
    binary_res = run_sparse_binary_experiment(
        sparsity_p=sparsity_p,
        repr_dim=repr_dim,
        batch_size=batch_size,
        epochs=epochs,
        eval_every=10,
        lr=lr,
        temperature=temperature,
        seed=seed,
        device=device,
    )

    combined = {
        "gmm": gmm_res,
        "binary": binary_res,
    }
    save_json(combined, out / "gmm_binary_results.json")
    print(f"[Synthetic GMM & Binary] Saved results to {out}")
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Synthetic GMM & Sparse Binary experiments.")
    parser.add_argument("--output-dir", type=str, default=str(ROOT_DIR / "results" / "synthetic_gmm_binary"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--repr-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--mean-std", type=float, default=0.05)
    parser.add_argument("--sparsity-p", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_additional_synthetic_experiments(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        repr_dim=args.repr_dim,
        lr=args.lr,
        temperature=args.temperature,
        mean_std=args.mean_std,
        sparsity_p=args.sparsity_p,
        seed=args.seed,
        device_str=args.device,
    )

