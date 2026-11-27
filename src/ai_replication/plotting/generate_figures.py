"""Generate all figures (Figs. 2, 3, 4, 6, 7, 8, 9, 10, 11) and Tables 1 & 2 from saved results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common.metrics import AD_THRESHOLD
from common.utils import ensure_dir, load_json


def _plot_fig2_alignment_uniformity_violins(
    laplace_json: dict[str, Any],
    laplace_npz: Any,
    out_dir: Path,
) -> Path:
    """Reproduce Figure 2: Uniformity vs. alignment violin plots across dimensions & batch sizes."""
    dims = sorted(laplace_json["dims"])
    batch_sizes = sorted(laplace_json["batch_sizes"])
    fixed_bs = max(batch_sizes)
    fixed_dim = max(dims)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharey="row")
    fig.suptitle("Figure 2: Uniformity vs. Alignment Across Settings (Synthetic Laplace)", fontsize=12)

    # Left column: sweep Dimensions at fixed max batch size
    neg_by_dim = [laplace_npz[f"d{d}_bs{fixed_bs}_neg_cos"] for d in dims]
    pos_by_dim = [laplace_npz[f"d{d}_bs{fixed_bs}_pos_cos"] for d in dims]
    x_dims = np.arange(1, len(dims) + 1)

    vp_u1 = axes[0, 0].violinplot(neg_by_dim, positions=x_dims, showmeans=True, showextrema=False)
    for body in vp_u1["bodies"]:
        body.set_facecolor("#4c72b0")
        body.set_alpha(0.7)
    axes[0, 0].axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    axes[0, 0].set_ylabel("Cosine similarity\n(Uniformity / Negatives)")
    axes[0, 0].set_title(f"Varying Dimension (Batch size = {fixed_bs})")
    axes[0, 0].set_xticks(x_dims)
    axes[0, 0].set_xticklabels([str(d) for d in dims])

    vp_a1 = axes[1, 0].violinplot(pos_by_dim, positions=x_dims, showmeans=True, showextrema=False)
    for body in vp_a1["bodies"]:
        body.set_facecolor("#55a868")
        body.set_alpha(0.7)
    axes[1, 0].set_ylabel("Cosine similarity\n(Alignment / Positives)")
    axes[1, 0].set_xlabel("Dimensions")
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].set_xticks(x_dims)
    axes[1, 0].set_xticklabels([str(d) for d in dims])

    # Right column: sweep Batch sizes at fixed max dimension
    neg_by_bs = [laplace_npz[f"d{fixed_dim}_bs{bs}_neg_cos"] for bs in batch_sizes]
    pos_by_bs = [laplace_npz[f"d{fixed_dim}_bs{bs}_pos_cos"] for bs in batch_sizes]
    x_bs = np.arange(1, len(batch_sizes) + 1)

    vp_u2 = axes[0, 1].violinplot(neg_by_bs, positions=x_bs, showmeans=True, showextrema=False)
    for body in vp_u2["bodies"]:
        body.set_facecolor("#4c72b0")
        body.set_alpha(0.7)
    axes[0, 1].axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    axes[0, 1].set_title(f"Varying Batch Size (Dim = {fixed_dim})")
    axes[0, 1].set_xticks(x_bs)
    axes[0, 1].set_xticklabels([str(bs) for bs in batch_sizes])

    vp_a2 = axes[1, 1].violinplot(pos_by_bs, positions=x_bs, showmeans=True, showextrema=False)
    for body in vp_a2["bodies"]:
        body.set_facecolor("#55a868")
        body.set_alpha(0.7)
    axes[1, 1].set_xlabel("Batch size")
    axes[1, 1].set_ylim(0.0, 1.05)
    axes[1, 1].set_xticks(x_bs)
    axes[1, 1].set_xticklabels([str(bs) for bs in batch_sizes])

    fig.tight_layout()
    target = out_dir / "fig2_uniformity_vs_alignment.png"
    fig.savefig(target, dpi=200)
    plt.close(fig)
    return target


def _plot_fig3_synthetic_diagnostics(
    laplace_json: dict[str, Any],
    laplace_npz: Any,
    out_dir: Path,
) -> Path:
    """Reproduce Figure 3: Norm stats, norm histograms, and AD/DP diagnostics vs. batch size."""
    dims = sorted(laplace_json["dims"])
    batch_sizes = sorted(laplace_json["batch_sizes"])
    lookup = {
        (r["config"]["repr_dim"], r["config"]["batch_size"]): r
        for r in laplace_json["sweep"]
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle("Figure 3: Synthetic Laplace Data Experiments (Thin-Shell & Normality)", fontsize=12)

    # Top-left: Norms mean vs Batch size
    # Bottom-left: CV vs Batch size
    # Bottom-middle: AD Value vs Batch size
    # Bottom-right: DP Value vs Batch size
    for d in dims:
        means = [lookup[(d, bs)]["diagnostics"]["unnormalized"]["norm_mean"] for bs in batch_sizes]
        cvs = [lookup[(d, bs)]["diagnostics"]["unnormalized"]["cv"] for bs in batch_sizes]
        ads = [lookup[(d, bs)]["diagnostics"]["unnormalized"]["ad_avg"] for bs in batch_sizes]
        dps = [lookup[(d, bs)]["diagnostics"]["unnormalized"]["dp_avg"] for bs in batch_sizes]

        axes[0, 0].plot(batch_sizes, means, marker="o", markersize=3, label=f"d={d}")
        axes[1, 0].plot(batch_sizes, cvs, marker="o", markersize=3, label=f"d={d}")
        axes[1, 1].plot(batch_sizes, ads, marker="o", markersize=3, label=f"d={d}")
        axes[1, 2].plot(batch_sizes, dps, marker="o", markersize=3, label=f"d={d}")

    axes[0, 0].set_ylabel("Norms mean")
    axes[0, 0].set_xlabel("Batch size")
    axes[0, 0].legend(title="Dimensions", fontsize=8)
    axes[0, 0].grid(alpha=0.3)

    axes[1, 0].set_ylabel("CV")
    axes[1, 0].set_xlabel("Batch size")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].axhline(AD_THRESHOLD, color="red", linestyle="--", linewidth=0.9, label="AD=0.752")
    axes[1, 1].set_ylabel("AD Value")
    axes[1, 1].set_xlabel("Batch size")
    axes[1, 1].grid(alpha=0.3)

    axes[1, 2].axhline(0.05, color="red", linestyle="--", linewidth=0.9, label="p=0.05")
    axes[1, 2].set_ylabel("DP Value (mean p)")
    axes[1, 2].set_xlabel("Batch size")
    axes[1, 2].legend(title="Dimensions", fontsize=8)
    axes[1, 2].grid(alpha=0.3)

    # Top-middle & Top-right: Norm histograms for Dim=32 (or min dim) and Dim=128 (or mid/max dim)
    hist_dims = [
        32 if 32 in dims else dims[0],
        128 if 128 in dims else dims[min(len(dims) - 1, 2)],
    ]
    hist_bs = [b for b in [8, 32, 64, 128] if b in batch_sizes] or batch_sizes[:4]
    for ax_idx, h_dim in enumerate(hist_dims):
        ax = axes[0, ax_idx + 1]
        for bs in hist_bs:
            norms = laplace_npz[f"d{h_dim}_bs{bs}_norms"]
            ax.hist(norms, bins=40, density=True, alpha=0.5, label=f"N={bs}")
        ax.set_title(f"Dim = {h_dim}")
        ax.set_xlabel("Norm value")
        ax.set_ylabel("Density")
        ax.legend(title="Batch size", fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    target = out_dir / "fig3_synthetic_diagnostics.png"
    fig.savefig(target, dpi=200)
    plt.close(fig)
    return target


def _plot_fig7_fig8_histograms(
    laplace_json: dict[str, Any],
    laplace_npz: Any,
    out_dir: Path,
) -> tuple[Path, Path]:
    """Reproduce Figures 7 and 8: Histogram views of alignment and uniformity."""
    dims = sorted(laplace_json["dims"])
    batch_sizes = sorted(laplace_json["batch_sizes"])
    max_d = max(dims)
    min_d = min(dims)
    max_bs = max(batch_sizes)
    min_bs = min(batch_sizes)

    # Figure 7: Alignment & uniformity vs. batch size
    fig7, axes7 = plt.subplots(1, 3, figsize=(14, 4))
    fig7.suptitle("Figure 7: Alignment and Uniformity vs. Batch Size", fontsize=12)
    for bs in batch_sizes:
        neg = laplace_npz[f"d{max_d}_bs{bs}_neg_cos"]
        pos = laplace_npz[f"d{max_d}_bs{bs}_pos_cos"]
        axes7[0].hist(neg, bins=60, range=(-0.6, 1.0), weights=np.ones_like(neg) / len(neg), alpha=0.45, label=f"N={bs}")
        axes7[0].hist(pos, bins=60, range=(-0.6, 1.0), weights=np.ones_like(pos) / len(pos), alpha=0.35)
        axes7[1].hist(neg, bins=60, range=(-0.6, 0.6), weights=np.ones_like(neg) / len(neg), alpha=0.45, label=f"N={bs}")
        neg_small = laplace_npz[f"d{min_d}_bs{bs}_neg_cos"]
        axes7[2].hist(neg_small, bins=60, range=(-0.6, 0.6), weights=np.ones_like(neg_small) / len(neg_small), alpha=0.45, label=f"N={bs}")

    axes7[0].set_title(f"Dim = {max_d} (Full)")
    axes7[1].set_title(f"Dim = {max_d} (Negatives Zoom)")
    axes7[2].set_title(f"Dim = {min_d} (Negatives Zoom)")
    for ax in axes7:
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Relative frequency")
        ax.grid(alpha=0.3)
    axes7[2].legend(title="Batch size", fontsize=8)
    fig7.tight_layout()
    target7 = out_dir / "fig7_alignment_uniformity_vs_batch.png"
    fig7.savefig(target7, dpi=200)
    plt.close(fig7)

    # Figure 8: Alignment & uniformity vs. dimensionality
    fig8, axes8 = plt.subplots(1, 3, figsize=(14, 4))
    fig8.suptitle("Figure 8: Alignment and Uniformity vs. Dimensionality", fontsize=12)
    for d in dims:
        neg = laplace_npz[f"d{d}_bs{max_bs}_neg_cos"]
        pos = laplace_npz[f"d{d}_bs{max_bs}_pos_cos"]
        axes8[0].hist(neg, bins=60, range=(-0.6, 1.0), weights=np.ones_like(neg) / len(neg), alpha=0.45, label=f"d={d}")
        axes8[0].hist(pos, bins=60, range=(-0.6, 1.0), weights=np.ones_like(pos) / len(pos), alpha=0.35)
        axes8[1].hist(neg, bins=60, range=(-0.6, 0.6), weights=np.ones_like(neg) / len(neg), alpha=0.45, label=f"d={d}")
        neg_small = laplace_npz[f"d{d}_bs{min_bs}_neg_cos"]
        axes8[2].hist(neg_small, bins=60, range=(-0.6, 0.6), weights=np.ones_like(neg_small) / len(neg_small), alpha=0.45, label=f"d={d}")

    axes8[0].set_title(f"Batch size = {max_bs} (Full)")
    axes8[1].set_title(f"Batch size = {max_bs} (Negatives Zoom)")
    axes8[2].set_title(f"Batch size = {min_bs} (Negatives Zoom)")
    for ax in axes8:
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Relative frequency")
        ax.grid(alpha=0.3)
    axes8[2].legend(title="Dimensions", fontsize=8)
    fig8.tight_layout()
    target8 = out_dir / "fig8_alignment_uniformity_vs_dim.png"
    fig8.savefig(target8, dpi=200)
    plt.close(fig8)

    return target7, target8


def _plot_fig11_pushforward(
    laplace_json: dict[str, Any],
    laplace_npz: Any,
    out_dir: Path,
) -> Path:
    """Reproduce Figure 11: Encoder pushforward log-likelihood correlation across A values."""
    pf_records = laplace_json.get("pushforward", [])
    n_panels = max(len(pf_records), 1)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]
    fig.suptitle("Figure 11: Encoder 'Pushforward' Log-Likelihood Correlation", fontsize=12)

    for ax, rec in zip(axes, pf_records):
        a_val = rec["corr_a"]
        tag = str(a_val).replace(".", "p")
        log_p_data = laplace_npz[f"pf_A{tag}_log_p_data"]
        log_p_emb = laplace_npz[f"pf_A{tag}_log_p_emb"]
        ax.scatter(log_p_emb, log_p_data, s=4, alpha=0.35, color="#1f77b4")

        xs = np.linspace(np.min(log_p_emb), np.max(log_p_emb), 100)
        ys = rec["fit_slope"] * xs + rec["fit_intercept"]
        ax.plot(xs, ys, color="red", linewidth=1.5)

        ax.set_title(f"Pearson = {rec['pearson_r']:.2f}")
        ax.set_xlabel("Log-P(embeddings)")
        ax.set_ylabel("Log-P(data)")
        ax.text(
            0.65,
            0.08,
            f"$\\rho = {a_val:.1f}$",
            transform=ax.transAxes,
            fontsize=11,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )
        ax.grid(alpha=0.3)

    fig.tight_layout()
    target = out_dir / "fig11_encoder_pushforward.png"
    fig.savefig(target, dpi=200)
    plt.close(fig)
    return target


def _plot_fig4_cifar10_mlp(mlp_json: dict[str, Any], out_dir: Path) -> Path:
    """Reproduce Figure 4: CIFAR-10 two-layer MLP training dynamics (CV, AD, DP %)."""
    history = mlp_json["history"]
    epochs = [h["epoch"] for h in history]
    cvs = [h["diagnostics"]["unnormalized"]["cv"] for h in history]
    ads = [h["diagnostics"]["unnormalized"]["ad_avg"] for h in history]
    dp_fracs = [h["diagnostics"]["unnormalized"]["dp_norm_feat_pct"] / 100.0 for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    fig.suptitle("Figure 4: CIFAR-10 Two-Layer MLP Training Dynamics", fontsize=12)

    axes[0].plot(epochs, cvs, color="#1f77b4", linewidth=2)
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("CV")
    axes[0].set_title("Norm Concentration (CV)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, ads, color="#1f77b4", linewidth=2)
    axes[1].axhline(AD_THRESHOLD, color="red", linestyle="--", linewidth=1.2)
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("AD value")
    axes[1].set_title("Anderson-Darling Statistic")
    axes[1].text(epochs[-1] * 0.55, AD_THRESHOLD * 1.05, "Not Gaussian", color="red", fontsize=9)
    axes[1].text(epochs[-1] * 0.55, AD_THRESHOLD * 0.88, "Gaussian", color="green", fontsize=9)
    axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, dp_fracs, color="#d95f02", linewidth=2)
    axes[2].set_xlabel("Epochs")
    axes[2].set_ylabel("DP [%] Features (fraction)")
    axes[2].set_title("D'Agostino-Pearson Compliance")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    target = out_dir / "fig4_cifar10_mlp_dynamics.png"
    fig.savefig(target, dpi=200)
    plt.close(fig)
    return target


def _plot_pretrained_figures(
    pretrained_json: dict[str, Any],
    pretrained_npz: Any,
    out_dir: Path,
) -> list[Path]:
    """Reproduce Figures 6, 9, and 10 for pretrained models."""
    generated: list[Path] = []
    rec_by_key = {r["key"]: r for r in pretrained_json.get("results", [])}

    # Figure 6: Thin-shell radius histograms across pretrained models
    fig6_order = [
        ("densenet_coco", "DenseNet", "#2ca02c"),
        ("resnet34_coco", "ResNet-34", "#1f77b4"),
        ("clip_text_coco", "CLIP: Text", "#d62728"),
        ("clip_image_coco", "CLIP: Image", "#ff7f0e"),
        ("dino_coco", "DINO", "#9467bd"),
    ]
    present6 = [item for item in fig6_order if item[0] in rec_by_key and f"{item[0]}_norms" in pretrained_npz]
    if present6:
        fig, axes = plt.subplots(1, len(present6), figsize=(3.2 * len(present6), 3.2))
        if len(present6) == 1:
            axes = [axes]
        fig.suptitle("Figure 6: Thin-Shell Concentration Across Pretrained Models", fontsize=12)
        for ax, (key, title, color) in zip(axes, present6):
            norms = pretrained_npz[f"{key}_norms"]
            cv = rec_by_key[key]["diagnostics"]["unnormalized"]["cv"]
            ax.hist(norms, bins=40, density=True, color=color, alpha=0.75)
            ax.set_title(f"{title}\nCV = {cv:.3f}")
            ax.set_xlabel("Norm")
            ax.set_ylabel("Density")
            ax.grid(alpha=0.3)
        fig.tight_layout()
        p6 = out_dir / "fig6_pretrained_thin_shell.png"
        fig.savefig(p6, dpi=200)
        plt.close(fig)
        generated.append(p6)

    # Figures 9 & 10: Whitening & Uniformity for CLIP Image, CLIP Text, DINO
    white_models = [
        ("clip_image_coco", "CLIP: Image"),
        ("clip_text_coco", "CLIP: Text"),
        ("dino_coco", "DINO"),
    ]
    present_w = [
        item for item in white_models if f"{item[0]}_unnorm_raw_cos" in pretrained_npz
    ]
    if present_w:
        # Figure 9: Unnormalized representations
        fig9, axes9 = plt.subplots(1, len(present_w), figsize=(4.2 * len(present_w), 3.6))
        if len(present_w) == 1:
            axes9 = [axes9]
        fig9.suptitle("Figure 9: Whitening and Uniformity (Unnormalized Representations)", fontsize=12)
        for ax, (key, title) in zip(axes9, present_w):
            raw = pretrained_npz[f"{key}_unnorm_raw_cos"]
            white = pretrained_npz[f"{key}_unnorm_white_cos"]
            ax.hist(raw, bins=60, range=(-0.6, 1.0), weights=np.ones_like(raw) / len(raw), alpha=0.55, label="Raw")
            ax.hist(white, bins=60, range=(-0.6, 1.0), weights=np.ones_like(white) / len(white), alpha=0.55, label="Whitened")
            ax.set_title(title)
            ax.set_xlabel("Cosine similarity")
            ax.set_ylabel("Density (relative count)")
            ax.legend()
            ax.grid(alpha=0.3)
        fig9.tight_layout()
        p9 = out_dir / "fig9_whitening_unnormalized.png"
        fig9.savefig(p9, dpi=200)
        plt.close(fig9)
        generated.append(p9)

        # Figure 10: Normalized representations
        fig10, axes10 = plt.subplots(1, len(present_w), figsize=(4.2 * len(present_w), 3.6))
        if len(present_w) == 1:
            axes10 = [axes10]
        fig10.suptitle("Figure 10: Whitening and Uniformity (Normalized Representations)", fontsize=12)
        for ax, (key, title) in zip(axes10, present_w):
            raw = pretrained_npz[f"{key}_norm_raw_cos"]
            white = pretrained_npz[f"{key}_norm_white_cos"]
            ax.hist(raw, bins=60, range=(-0.6, 1.0), alpha=0.55, label="Raw")
            ax.hist(white, bins=60, range=(-0.6, 1.0), alpha=0.55, label="Whitened")
            ax.set_title(title)
            ax.set_xlabel("Cosine similarity")
            ax.set_ylabel("Count")
            ax.legend()
            ax.grid(alpha=0.3)
        fig10.tight_layout()
        p10 = out_dir / "fig10_whitening_normalized.png"
        fig10.savefig(p10, dpi=200)
        plt.close(fig10)
        generated.append(p10)

    return generated


def _fmt_diag_col(diag: dict[str, Any] | None) -> dict[str, str]:
    if diag is None:
        return {"cv": "-", "ad_avg": "-", "ad_pct": "-", "dp_avg": "-", "dp_pct": "-", "gauss": "-"}
    return {
        "cv": f"{diag['cv']:.3f}",
        "ad_avg": f"{diag['ad_avg']:.3f}",
        "ad_pct": f"{diag['ad_norm_feat_pct']:.1f}%",
        "dp_avg": f"{diag['dp_avg']:.3f}",
        "dp_pct": f"{diag['dp_norm_feat_pct']:.1f}%",
        "gauss": "Yes" if diag["is_gaussian"] else "No",
    }


def generate_tables_markdown(results_root: Path, out_dir: Path) -> Path:
    """Format reproduced Table 1 and Table 2 into a clean Markdown report."""
    lines: list[str] = ["# Reproduced Diagnostics Tables (Tables 1 & 2)\n"]

    # Collect Table 1 entries (unnormalized embeddings, matching paper Table 1 caption)
    t1_cols: dict[str, dict[str, Any] | None] = {
        "Laplace": None,
        "GMM": None,
        "Binary E0": None,
        "Binary E50": None,
        "Binary E100": None,
        "CIFAR-10 Supervised": None,
        "CIFAR-10 Contrastive": None,
    }

    lap_path = results_root / "synthetic_laplace" / "laplace_results.json"
    if lap_path.exists():
        lap_data = load_json(lap_path)
        t1_cols["Laplace"] = lap_data["table1_representative"]["diagnostics"]["unnormalized"]

    gb_path = results_root / "synthetic_gmm_binary" / "gmm_binary_results.json"
    if gb_path.exists():
        gb_data = load_json(gb_path)
        t1_cols["GMM"] = gb_data["gmm"]["final"]["unnormalized"]
        b_evals = gb_data["binary"]["epochs_eval"]
        if "epoch_0" in b_evals:
            t1_cols["Binary E0"] = b_evals["epoch_0"]["unnormalized"]
        if "epoch_50" in b_evals:
            t1_cols["Binary E50"] = b_evals["epoch_50"]["unnormalized"]
        if "epoch_100" in b_evals:
            t1_cols["Binary E100"] = b_evals["epoch_100"]["unnormalized"]

    rn_path = results_root / "cifar10_resnet18" / "cifar10_resnet18_results.json"
    if rn_path.exists():
        rn_data = load_json(rn_path)
        t1_cols["CIFAR-10 Supervised"] = rn_data["supervised"]["final"]["unnormalized"]
        t1_cols["CIFAR-10 Contrastive"] = rn_data["contrastive"]["final"]["unnormalized"]

    fmt = {k: _fmt_diag_col(v) for k, v in t1_cols.items()}
    headers = list(t1_cols.keys())

    lines.append("## Table 1: Gaussianity Diagnostics Across Data and Training Settings (Unnormalized)\n")
    lines.append("| Metric | " + " | ".join(headers) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(headers)) + "|")
    lines.append("| CV | " + " | ".join(fmt[h]["cv"] for h in headers) + " |")
    lines.append("| AD Avg. (< 0.752) | " + " | ".join(fmt[h]["ad_avg"] for h in headers) + " |")
    lines.append("| AD Norm. Feat. | " + " | ".join(fmt[h]["ad_pct"] for h in headers) + " |")
    lines.append("| DP Avg. (> 0.05) | " + " | ".join(fmt[h]["dp_avg"] for h in headers) + " |")
    lines.append("| DP Norm. Feat. | " + " | ".join(fmt[h]["dp_pct"] for h in headers) + " |")
    lines.append("| Gaussian? | " + " | ".join(fmt[h]["gauss"] for h in headers) + " |")
    lines.append("")

    # Table 2: Pretrained Models (Unnormalized / Normalized)
    pre_path = results_root / "pretrained" / "pretrained_results.json"
    lines.append("## Table 2: Gaussianity Diagnostics for Pretrained Models (Unnormalized / Normalized)\n")
    if pre_path.exists():
        pre_data = load_json(pre_path)
        lines.append(
            "| Training | Model (Test data) | CV | AD Avg. (< 0.752) | AD Norm. Feat. (%) | "
            "DP Avg. (> 0.05) | DP Norm. Feat. (%) | Gaussian? |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|:---:|")
        for rec in pre_data.get("results", []):
            un = rec["diagnostics"]["unnormalized"]
            nm = rec["diagnostics"]["normalized"]
            lines.append(
                f"| {rec['training']} | {rec['key']} ({rec['dataset']}) | {un['cv']:.4f} | "
                f"{un['ad_avg']:.3f} / {nm['ad_avg']:.3f} | "
                f"{un['ad_norm_feat_pct']:.1f}% / {nm['ad_norm_feat_pct']:.1f}% | "
                f"{un['dp_avg']:.4f} / {nm['dp_avg']:.4f} | "
                f"{un['dp_norm_feat_pct']:.1f}% / {nm['dp_norm_feat_pct']:.1f}% | "
                f"{'Yes' if un['is_gaussian'] else 'No'} |"
            )
    else:
        lines.append("_Pretrained diagnostics have not been executed yet._")
    lines.append("")

    target = out_dir / "tables_summary.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def generate_all_outputs(results_root: str | Path, figures_dir: str | Path) -> list[Path]:
    """Generate every figure and summary table for which result files exist."""
    res_root = Path(results_root)
    out = ensure_dir(figures_dir)
    generated: list[Path] = []

    # Synthetic Laplace (Figs 2, 3, 7, 8, 11)
    lap_json_path = res_root / "synthetic_laplace" / "laplace_results.json"
    lap_npz_path = res_root / "synthetic_laplace" / "laplace_distributions.npz"
    if lap_json_path.exists() and lap_npz_path.exists():
        lap_json = load_json(lap_json_path)
        with np.load(lap_npz_path) as lap_npz:
            generated.append(_plot_fig2_alignment_uniformity_violins(lap_json, lap_npz, out))
            generated.append(_plot_fig3_synthetic_diagnostics(lap_json, lap_npz, out))
            p7, p8 = _plot_fig7_fig8_histograms(lap_json, lap_npz, out)
            generated.extend([p7, p8])
            if lap_json.get("pushforward"):
                generated.append(_plot_fig11_pushforward(lap_json, lap_npz, out))

    # CIFAR-10 MLP dynamics (Fig 4)
    mlp_json_path = res_root / "cifar10_mlp" / "cifar10_mlp_results.json"
    if mlp_json_path.exists():
        generated.append(_plot_fig4_cifar10_mlp(load_json(mlp_json_path), out))

    # Pretrained models (Figs 6, 9, 10)
    pre_json_path = res_root / "pretrained" / "pretrained_results.json"
    pre_npz_path = res_root / "pretrained" / "pretrained_distributions.npz"
    if pre_json_path.exists() and pre_npz_path.exists():
        with np.load(pre_npz_path) as pre_npz:
            generated.extend(_plot_pretrained_figures(load_json(pre_json_path), pre_npz, out))

    # Tables 1 & 2 Markdown summary
    generated.append(generate_tables_markdown(res_root, out))

    print("[Plotting] Generated outputs:")
    for p in generated:
        print(f"  - {p}")
    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper figures and Tables 1 & 2.")
    parser.add_argument("--results-dir", type=str, default=str(ROOT_DIR / "results"))
    parser.add_argument("--figures-dir", type=str, default=str(ROOT_DIR / "results" / "figures"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_all_outputs(args.results_dir, args.figures_dir)

