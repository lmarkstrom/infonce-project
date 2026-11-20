"""End-to-end verification script for all PyTorch modules, training loops, metrics, and plots."""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common.datasets import (
    ImageNetRDomainDataset,
    MSCOCOValDataset,
    TwoCropTransform,
    build_simclr_cifar10_transform,
)
from common.losses import InfoNCELoss, SimCLRNTXentLoss, compute_population_infonce_terms
from common.metrics import (
    compute_negative_cosines,
    evaluate_embeddings_both,
    zca_whiten,
)
from common.models import TwoLayerMLPEncoder, build_matched_resnet18_pair
from common.utils import get_device, seed_everything
from experiments.run_synthetic_gmm_binary import run_additional_synthetic_experiments
from experiments.run_synthetic_laplace import run_laplace_experiments
from plotting.generate_figures import generate_all_outputs


def verify_losses_and_metrics() -> None:
    seed_everything(0)
    z1 = torch.randn(32, 64, requires_grad=True)
    z2 = torch.randn(32, 64, requires_grad=True)

    loss_infonce = InfoNCELoss(temperature=0.1)(z1, z2)
    loss_sym = InfoNCELoss(temperature=0.1, symmetric=True)(z1, z2)
    loss_ntxent = SimCLRNTXentLoss(temperature=0.1)(z1, z2)
    assert torch.isfinite(loss_infonce) and torch.isfinite(loss_sym) and torch.isfinite(loss_ntxent)
    loss_ntxent.backward()
    assert z1.grad is not None and torch.isfinite(z1.grad).all()

    pop_terms = compute_population_infonce_terms(z1.detach(), z2.detach(), temperature=0.1)
    assert all(np.isfinite(v) for v in pop_terms.values())

    # Check Gaussianity metrics on true isotropic Gaussian vs. heavy-tailed/exponential data
    rng = np.random.default_rng(42)
    g_data = rng.standard_normal((2000, 64))
    g_diag = evaluate_embeddings_both(g_data)
    assert g_diag["unnormalized"]["ad_avg"] < 0.752
    assert g_diag["unnormalized"]["dp_avg"] > 0.05
    assert g_diag["unnormalized"]["is_gaussian"] is True

    exp_data = rng.exponential(scale=1.0, size=(2000, 64))
    e_diag = evaluate_embeddings_both(exp_data)
    assert e_diag["unnormalized"]["ad_avg"] > 0.752
    assert e_diag["unnormalized"]["is_gaussian"] is False

    # Verify ZCA whitening produces near-identity covariance
    aniso = g_data @ (rng.standard_normal((64, 64)) * 0.5 + np.eye(64))
    white = zca_whiten(aniso)
    cov_white = np.cov(white, rowvar=False)
    assert np.allclose(cov_white, np.eye(64), atol=1e-2)
    cos_raw = compute_negative_cosines(aniso, max_pairs=5000)
    cos_white = compute_negative_cosines(white, max_pairs=5000)
    assert cos_raw.shape == cos_white.shape == (5000,)
    print("[OK] Losses, metrics, and ZCA whitening verified.")


def verify_vision_models_and_transforms() -> None:
    seed_everything(0)
    # Two-layer MLP forward/backward
    mlp = TwoLayerMLPEncoder(input_dim=3 * 32 * 32, hidden_dim=512, output_dim=256)
    x = torch.randn(16, 3, 32, 32)
    z = mlp(x)
    assert z.shape == (16, 256)

    # SimCLR CIFAR-10 transform on PIL image
    pil_img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
    two_crop = TwoCropTransform(build_simclr_cifar10_transform())
    v1, v2 = two_crop(pil_img)
    assert v1.shape == (3, 32, 32) and v2.shape == (3, 32, 32)

    # Matched ResNet-18 pair (using random init here so offline check needs no network)
    c_model, s_model = build_matched_resnet18_pair(pretrained=False, cifar_stem=True)
    for p1, p2 in zip(c_model.backbone.parameters(), s_model.backbone.parameters()):
        assert torch.equal(p1.data, p2.data)
    hc, zc = c_model(x)
    hs, logits = s_model(x)
    assert hc.shape == (16, 512) and zc.shape == (16, 128)
    assert hs.shape == (16, 512) and logits.shape == (16, 10)
    assert torch.allclose(hc, hs, atol=1e-6)
    print("[OK] CIFAR-10 MLP, SimCLR transforms, and matched ResNet-18 pair verified.")


def verify_pretrained_dataset_loaders(tmp_root: Path) -> None:
    # Verify MS-COCO and ImageNet-R Dataset classes on temporary image fixtures
    coco_dir = tmp_root / "val2017"
    coco_dir.mkdir(parents=True, exist_ok=True)
    for i in range(4):
        Image.fromarray(np.full((64, 64, 3), i * 50, dtype=np.uint8)).save(coco_dir / f"{i:012d}.jpg")
    ann_path = tmp_root / "captions_val2017.json"
    ann_path.write_text(
        '{"images":[{"id":0,"file_name":"000000000000.jpg"},{"id":1,"file_name":"000000000001.jpg"}],'
        '"annotations":[{"image_id":0,"caption":"a cat"},{"image_id":1,"caption":"a dog"}]}',
        encoding="utf-8",
    )
    coco_ds = MSCOCOValDataset(coco_dir, ann_path)
    assert len(coco_ds) == 2 and coco_ds[0][1] == "a cat"

    inr_dir = tmp_root / "imagenet-r" / "n01440764"
    inr_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(inr_dir / "sketch_0.jpg")
    Image.fromarray(np.ones((64, 64, 3), dtype=np.uint8)).save(inr_dir / "painting_0.jpg")
    sketch_ds = ImageNetRDomainDataset(tmp_root / "imagenet-r", domain="sketch")
    paint_ds = ImageNetRDomainDataset(tmp_root / "imagenet-r", domain="painting")
    assert len(sketch_ds) == 1 and len(paint_ds) == 1
    print("[OK] MS-COCO and ImageNet-R dataset loaders verified.")


def verify_synthetic_training_and_plotting(tmp_root: Path, device_str: str = "cuda") -> None:
    print(f"[Info] Running end-to-end synthetic verification on device={get_device(device_str)}...")
    res_root = tmp_root / "results"

    lap_res = run_laplace_experiments(
        output_dir=res_root / "synthetic_laplace",
        dims=[32, 128],
        batch_sizes=[8, 64],
        epochs=15,
        pushforward_as=[0.2, 0.5, 0.8],
        pushforward_d_data=512,
        pushforward_dim=512,
        pushforward_batch_size=64,
        n_train=4000,
        n_test=1000,
        seed=42,
        device_str=device_str,
    )
    rep = lap_res["table1_representative"]["diagnostics"]["unnormalized"]
    assert rep["cv"] < 0.20, f"Unexpectedly high CV on Laplace: {rep['cv']}"
    assert rep["ad_avg"] < 0.752, f"Laplace AD failed Gaussian check: {rep['ad_avg']}"
    for pf in lap_res["pushforward"]:
        assert pf["pearson_r"] > 0.80, f"Unexpectedly low pushforward Pearson r: {pf['pearson_r']}"

    gb_res = run_additional_synthetic_experiments(
        output_dir=res_root / "synthetic_gmm_binary",
        epochs=30,
        batch_size=128,
        repr_dim=256,
        seed=42,
        device_str=device_str,
    )
    gmm_un = gb_res["gmm"]["final"]["unnormalized"]
    assert gmm_un["cv"] < 0.15 and gmm_un["ad_avg"] < 0.752 and gmm_un["dp_avg"] > 0.05
    bin_e30 = gb_res["binary"]["epochs_eval"]["epoch_30"]["unnormalized"]
    assert bin_e30["cv"] < 0.15 and bin_e30["ad_avg"] < 0.752 and bin_e30["dp_avg"] > 0.05

    outputs = generate_all_outputs(res_root, res_root / "figures")
    assert any(p.name == "fig2_uniformity_vs_alignment.png" for p in outputs)
    assert any(p.name == "fig3_synthetic_diagnostics.png" for p in outputs)
    assert any(p.name == "fig11_encoder_pushforward.png" for p in outputs)
    assert any(p.name == "tables_summary.md" for p in outputs)
    print("[OK] End-to-end synthetic training, Gaussianity convergence, and figure/table generation verified.")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Verify InfoNCE reproduction modules end-to-end.")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    tmp_dir = Path(tempfile.mkdtemp(prefix="infonce_verify_"))
    try:
        verify_losses_and_metrics()
        verify_vision_models_and_transforms()
        verify_pretrained_dataset_loaders(tmp_dir)
        verify_synthetic_training_and_plotting(tmp_dir, device_str=args.device)
        print("\nALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

