"""Synthetic Laplace(0, 1) Experiments (Figures 2, 3, 7, 8, 11 & Table 1 - Laplace).

Reproduces Appendix E.1 'Synthetic Laplace data experiments' and Appendix E.2 'Encoder pushforward':
- Dataset: i.i.d. Laplace(0, 1), d_data = 1024, 20k train / 5k test
- Model: Linear encoder nn.Linear(1024, d)
- Augmentations: Eq. (53) Gaussian mixing (default A = 0.6; A in {0.2, 0.5, 0.8} for Fig. 11)
  followed by additive Gaussian noise (std 0.2), feature dropout (p = 0.1), and
  random multiplicative scaling exp(N(0, 0.1^2)).
- Sweep: d in {32, 64, 128, 256, 512}, batch size N in {8, 16, 32, 48, 64, 96, 128}, tau = 0.1.
- Optimizer: Adam, lr = 1e-3, 150 epochs (default).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common.datasets import augment_laplace_pair, generate_laplace_dataset
from common.losses import InfoNCELoss
from common.metrics import (
    compute_alignment_uniformity_stats,
    compute_norm_stats,
    compute_pushforward_likelihood_correlation,
    evaluate_embeddings_both,
)
from common.models import LinearEncoder
from common.utils import ensure_dir, get_device, save_json, seed_everything


def train_laplace_linear_encoder(
    x_train: torch.Tensor,
    x_test: torch.Tensor,
    repr_dim: int,
    batch_size: int,
    epochs: int = 150,
    lr: float = 1e-3,
    temperature: float = 0.1,
    corr_a: float = 0.6,
    seed: int = 42,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Train a single LinearEncoder with InfoNCE on synthetic Laplace(0, 1) data."""
    seed_everything(seed)
    device = device or get_device()

    d_data = x_train.shape[1]
    model = LinearEncoder(input_dim=d_data, output_dim=repr_dim).to(device)
    criterion = InfoNCELoss(temperature=temperature, symmetric=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x_train_dev = x_train.to(device)
    x_test_dev = x_test.to(device)
    n_train = x_train_dev.shape[0]
    n_batches = n_train // batch_size

    loss_history: list[float] = []
    model.train()
    for _epoch in range(1, epochs + 1):
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            xb = x_train_dev[idx]
            v1, v2 = augment_laplace_pair(xb, corr_a=corr_a)
            z1 = model(v1)
            z2 = model(v2)
            loss = criterion(z1, z2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        loss_history.append(epoch_loss / max(n_batches, 1))

    model.eval()
    with torch.no_grad():
        z_train = model(x_train_dev).cpu().numpy()
        z_test = model(x_test_dev).cpu().numpy()
        t1, t2 = augment_laplace_pair(x_test_dev, corr_a=corr_a)
        z_test_v1 = model(t1).cpu().numpy()
        z_test_v2 = model(t2).cpu().numpy()

    diag_both = evaluate_embeddings_both(z_test)
    norm_info = compute_norm_stats(z_test)
    align_unif = compute_alignment_uniformity_stats(
        z_test_v1, z_test_v2, temperature=temperature, seed=seed
    )

    return {
        "config": {
            "d_data": d_data,
            "repr_dim": repr_dim,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "temperature": temperature,
            "corr_a": corr_a,
            "seed": seed,
        },
        "final_train_loss": loss_history[-1],
        "loss_history": loss_history,
        "diagnostics": diag_both,
        "alignment_mean": align_unif["alignment_mean"],
        "alignment_std": align_unif["alignment_std"],
        "uniformity_neg_cos_mean": align_unif["uniformity_neg_cos_mean"],
        "uniformity_neg_cos_std": align_unif["uniformity_neg_cos_std"],
        "uniformity_phi": align_unif["uniformity_phi"],
        "population_infonce": align_unif["population_infonce"],
        "arrays": {
            "norms": norm_info["norms"].astype(np.float32),
            "pos_cos": align_unif["pos_cos"],
            "neg_cos": align_unif["neg_cos"],
            "z_train": z_train.astype(np.float32),
            "z_test": z_test.astype(np.float32),
        },
    }


def run_laplace_experiments(
    output_dir: str | Path,
    dims: list[int] | None = None,
    batch_sizes: list[int] | None = None,
    epochs: int = 150,
    lr: float = 1e-3,
    temperature: float = 0.1,
    corr_a: float = 0.6,
    pushforward_as: list[float] | None = None,
    pushforward_d_data: int = 512,
    pushforward_dim: int = 512,
    pushforward_batch_size: int = 128,
    n_train: int = 20_000,
    n_test: int = 5_000,
    d_data: int = 1024,
    seed: int = 42,
    device_str: str = "cuda",
) -> dict[str, Any]:
    """Run full synthetic Laplace dimension/batch-size sweep and pushforward experiment."""
    out = ensure_dir(output_dir)
    device = get_device(device_str)
    dims = dims or [32, 64, 128, 256, 512]
    batch_sizes = batch_sizes or [8, 16, 32, 48, 64, 96, 128]
    pushforward_as = pushforward_as or [0.2, 0.5, 0.8]

    print(
        f"[Synthetic Laplace] Device={device}, d_data={d_data}, "
        f"train={n_train}, test={n_test}, epochs={epochs}, tau={temperature}"
    )
    x_train, x_test = generate_laplace_dataset(
        n_train=n_train, n_test=n_test, d_data=d_data, seed=seed
    )

    sweep_records: list[dict[str, Any]] = []
    saved_arrays: dict[str, np.ndarray] = {}

    for d in dims:
        for bs in batch_sizes:
            run_res = train_laplace_linear_encoder(
                x_train=x_train,
                x_test=x_test,
                repr_dim=d,
                batch_size=bs,
                epochs=epochs,
                lr=lr,
                temperature=temperature,
                corr_a=corr_a,
                seed=seed,
                device=device,
            )
            un = run_res["diagnostics"]["unnormalized"]
            print(
                f"  d={d:3d}, N={bs:3d} | loss={run_res['final_train_loss']:.4f} | "
                f"CV={un['cv']:.4f} | AD={un['ad_avg']:.3f} ({un['ad_norm_feat_pct']:.1f}%) | "
                f"DP={un['dp_avg']:.3f} ({un['dp_norm_feat_pct']:.1f}%) | "
                f"align={run_res['alignment_mean']:.3f}, neg_std={run_res['uniformity_neg_cos_std']:.3f}"
            )
            record = {k: v for k, v in run_res.items() if k != "arrays"}
            sweep_records.append(record)

            # Persist arrays needed for Figures 2, 3, 7, 8
            key_prefix = f"d{d}_bs{bs}"
            saved_arrays[f"{key_prefix}_norms"] = run_res["arrays"]["norms"]
            # Subsample cosines up to 50k entries to keep NPZ compact while preserving histograms
            pos_c = run_res["arrays"]["pos_cos"]
            neg_c = run_res["arrays"]["neg_cos"]
            if neg_c.shape[0] > 50_000:
                neg_c = neg_c[:50_000]
            saved_arrays[f"{key_prefix}_pos_cos"] = pos_c
            saved_arrays[f"{key_prefix}_neg_cos"] = neg_c

    # Representative Table 1 Laplace entry (prefer d=256, N=128 if in sweep, else max d & max N)
    rep_d = 256 if 256 in dims else max(dims)
    rep_bs = 128 if 128 in batch_sizes else max(batch_sizes)
    table1_laplace = next(
        r
        for r in sweep_records
        if r["config"]["repr_dim"] == rep_d and r["config"]["batch_size"] == rep_bs
    )

    # Encoder Pushforward experiment (Figure 11): A in {0.2, 0.5, 0.8}
    # Uses matching input/output dimension (default 512 -> 512) matching Fig. 11 log-likelihood scale
    print(
        f"[Synthetic Laplace Pushforward] Running A in {pushforward_as} "
        f"(d_data={pushforward_d_data} -> d={pushforward_dim}, N={pushforward_batch_size})"
    )
    if pushforward_d_data == d_data:
        pf_x_train, pf_x_test = x_train, x_test
    else:
        pf_x_train, pf_x_test = generate_laplace_dataset(
            n_train=n_train, n_test=n_test, d_data=pushforward_d_data, seed=seed
        )
    pushforward_records: list[dict[str, Any]] = []
    pf_x_test_np = pf_x_test.numpy()
    for a_val in pushforward_as:
        pf_run = train_laplace_linear_encoder(
            x_train=pf_x_train,
            x_test=pf_x_test,
            repr_dim=pushforward_dim,
            batch_size=pushforward_batch_size,
            epochs=epochs,
            lr=lr,
            temperature=temperature,
            corr_a=a_val,
            seed=seed,
            device=device,
        )
        pf_corr = compute_pushforward_likelihood_correlation(
            x_test=pf_x_test_np,
            z_train=pf_run["arrays"]["z_train"],
            z_test=pf_run["arrays"]["z_test"],
        )
        print(
            f"  A={a_val:.1f} | Pearson r={pf_corr['pearson_r']:.4f} "
            f"(p={pf_corr['pearson_p']:.2e}) | CV={pf_run['diagnostics']['unnormalized']['cv']:.4f}"
        )
        pushforward_records.append(
            {
                "corr_a": a_val,
                "repr_dim": pushforward_dim,
                "batch_size": pushforward_batch_size,
                "pearson_r": pf_corr["pearson_r"],
                "pearson_p": pf_corr["pearson_p"],
                "fit_slope": pf_corr["fit_slope"],
                "fit_intercept": pf_corr["fit_intercept"],
                "diagnostics": pf_run["diagnostics"],
            }
        )
        tag = str(a_val).replace(".", "p")
        saved_arrays[f"pf_A{tag}_log_p_data"] = pf_corr["log_p_data"]
        saved_arrays[f"pf_A{tag}_log_p_emb"] = pf_corr["log_p_emb"]

    summary = {
        "dims": dims,
        "batch_sizes": batch_sizes,
        "epochs": epochs,
        "temperature": temperature,
        "corr_a": corr_a,
        "d_data": d_data,
        "n_train": n_train,
        "n_test": n_test,
        "table1_representative": table1_laplace,
        "sweep": sweep_records,
        "pushforward": pushforward_records,
    }
    save_json(summary, out / "laplace_results.json")
    np.savez_compressed(out / "laplace_distributions.npz", **saved_arrays)
    print(f"[Synthetic Laplace] Saved results to {out}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Synthetic Laplace InfoNCE experiments.")
    parser.add_argument("--output-dir", type=str, default=str(ROOT_DIR / "results" / "synthetic_laplace"))
    parser.add_argument("--dims", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 32, 48, 64, 96, 128])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--corr-a", type=float, default=0.6)
    parser.add_argument("--pushforward-as", type=float, nargs="+", default=[0.2, 0.5, 0.8])
    parser.add_argument("--pushforward-d-data", type=int, default=512)
    parser.add_argument("--pushforward-dim", type=int, default=512)
    parser.add_argument("--pushforward-batch-size", type=int, default=128)
    parser.add_argument("--n-train", type=int, default=20_000)
    parser.add_argument("--n-test", type=int, default=5_000)
    parser.add_argument("--d-data", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_laplace_experiments(
        output_dir=args.output_dir,
        dims=args.dims,
        batch_sizes=args.batch_sizes,
        epochs=args.epochs,
        lr=args.lr,
        temperature=args.temperature,
        corr_a=args.corr_a,
        pushforward_as=args.pushforward_as,
        pushforward_d_data=args.pushforward_d_data,
        pushforward_dim=args.pushforward_dim,
        pushforward_batch_size=args.pushforward_batch_size,
        n_train=args.n_train,
        n_test=args.n_test,
        d_data=args.d_data,
        seed=args.seed,
        device_str=args.device,
    )

