"""Pretrained Foundation & Supervised Model Diagnostics (Table 2, Figures 6, 9, 10).

Reproduces Section 5 'Pretrained models' and Appendix E.1/E.2:
- Supervised (ImageNet-pretrained):
  * ResNet-34 on MS-COCO val (5k images)
  * DenseNet (DenseNet-121) on MS-COCO val (5k images)
- Self-supervised:
  * DINO (ViT-B/32 or ViT-B/16) on MS-COCO val
  * CLIP (ViT-L/14) Image encoder on MS-COCO val
  * CLIP (ViT-L/14) Text encoder on MS-COCO val captions
  * CLIP (ViT-L/14) Image encoder on ImageNet-R (Sketch and Painting domains)
- Diagnostics:
  * Coordinate-wise AD (< 0.752) and DP (p > 0.05) for Unnormalized & Normalized embeddings (Table 2)
  * Representation norm histograms & CV (Figure 6)
  * Negative cosine similarity histograms before (Raw) and after ZCA whitening for
    unnormalized (Figure 9) and normalized (Figure 10) embeddings.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common.datasets import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ImageNetRDomainDataset,
    MSCOCOValDataset,
)
from common.metrics import (
    compute_negative_cosines,
    compute_norm_stats,
    evaluate_embeddings_both,
    l2_normalize_numpy,
    zca_whiten,
)
from common.utils import default_num_workers, ensure_dir, get_device, save_json, seed_everything


def _imagenet_eval_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class ResNet34FeatureExtractor(nn.Module):
    """ImageNet-pretrained ResNet-34 returning last-layer 512-dim embeddings before fc."""

    def __init__(self) -> None:
        super().__init__()
        backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class DenseNetFeatureExtractor(nn.Module):
    """ImageNet-pretrained DenseNet returning last-layer pooled embeddings before classifier."""

    def __init__(self, variant: str = "densenet121") -> None:
        super().__init__()
        if variant == "densenet121":
            model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        elif variant == "densenet161":
            model = models.densenet161(weights=models.DenseNet161_Weights.IMAGENET1K_V1)
        elif variant == "densenet201":
            model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported DenseNet variant: {variant}")
        self.features = model.features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = F.relu(out, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        return torch.flatten(out, 1)


def load_dino_model(
    device: torch.device,
    dino_model_name: str = "dino_vitb16",
) -> Tuple[nn.Module, Callable[[Any], torch.Tensor], str]:
    """Load pretrained DINO ViT backbone (supports timm, transformers, or torch.hub)."""
    backend_errors: list[str] = []

    # 1. Try timm if installed (supports ViT-B/32 & DINO checkpoints)
    try:
        import timm  # type: ignore

        timm_candidates = [
            "vit_base_patch32_224.dino",
            "vit_base_patch16_224.dino",
        ]
        for cand in timm_candidates:
            try:
                model = timm.create_model(cand, pretrained=True, num_classes=0).to(device).eval()
                data_cfg = timm.data.resolve_model_data_config(model)
                tf = timm.data.create_transform(**data_cfg, is_training=False)
                return model, tf, f"timm:{cand}"
            except Exception as exc:
                backend_errors.append(f"timm({cand}): {exc}")
    except ImportError as exc:
        backend_errors.append(f"timm unavailable: {exc}")

    # 2. Try HuggingFace transformers if installed
    try:
        from transformers import AutoImageProcessor, AutoModel  # type: ignore

        hf_name = "facebook/dino-vitb16"
        processor = AutoImageProcessor.from_pretrained(hf_name)
        hf_model = AutoModel.from_pretrained(hf_name).to(device).eval()

        class _HFDINOWrapper(nn.Module):
            def __init__(self, m: nn.Module) -> None:
                super().__init__()
                self.m = m

            def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
                outputs = self.m(pixel_values=pixel_values)
                return outputs.last_hidden_state[:, 0]

        def _tf(img: Any) -> torch.Tensor:
            return processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

        return _HFDINOWrapper(hf_model), _tf, f"transformers:{hf_name}"
    except ImportError as exc:
        backend_errors.append(f"transformers unavailable: {exc}")

    # 3. Native PyTorch Hub loader for official facebookresearch/dino
    hub_model = torch.hub.load("facebookresearch/dino:main", dino_model_name)
    hub_model = hub_model.to(device).eval()
    return hub_model, _imagenet_eval_transform(224), f"torch.hub:facebookresearch/dino:{dino_model_name}"


class CLIPWrapper:
    """Unified loader for pretrained CLIP ViT-L/14 (image and text encoders).

    Supports open_clip, OpenAI clip, and HuggingFace transformers backends and
    always returns UNNORMALIZED last-layer projected embeddings so both
    unnormalized and normalized diagnostics (Table 2, Figs. 6, 9, 10) can be computed.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.backend = ""
        self.preprocess: Callable[[Any], torch.Tensor]
        backend_errors: list[str] = []

        # 1. Try open_clip
        try:
            import open_clip  # type: ignore

            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained="openai", device=device
            )
            model.eval()
            self.model = model
            self.tokenizer = open_clip.get_tokenizer("ViT-L-14")
            self.preprocess = preprocess
            self.backend = "open_clip:ViT-L-14(openai)"
            return
        except ImportError as exc:
            backend_errors.append(f"open_clip unavailable: {exc}")

        # 2. Try OpenAI clip
        try:
            import clip  # type: ignore

            model, preprocess = clip.load("ViT-L/14", device=device)
            model.eval()
            self.model = model
            self.tokenizer = clip.tokenize
            self.preprocess = preprocess
            self.backend = "openai_clip:ViT-L/14"
            return
        except ImportError as exc:
            backend_errors.append(f"openai clip unavailable: {exc}")

        # 3. Try HuggingFace transformers
        try:
            from transformers import CLIPModel, CLIPProcessor  # type: ignore

            hf_name = "openai/clip-vit-large-patch14"
            self.hf_processor = CLIPProcessor.from_pretrained(hf_name)
            self.model = CLIPModel.from_pretrained(hf_name).to(device).eval()

            def _hf_img_tf(img: Any) -> torch.Tensor:
                return self.hf_processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

            self.preprocess = _hf_img_tf
            self.backend = f"transformers:{hf_name}"
            return
        except ImportError as exc:
            backend_errors.append(f"transformers unavailable: {exc}")
            raise ImportError(
                "Loading pretrained CLIP (ViT-L/14) requires one of: "
                "'open_clip_torch', 'clip' (OpenAI), or 'transformers'. "
                f"Tried backends: {'; '.join(backend_errors)}"
            ) from exc

    @torch.no_grad()
    def encode_images(self, loader: DataLoader) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for images, _meta in loader:
            images = images.to(self.device, non_blocking=True)
            if self.backend.startswith("transformers:"):
                feats = self.model.get_image_features(pixel_values=images)
            else:
                feats = self.model.encode_image(images)
            chunks.append(feats.float().cpu().numpy())
        return np.concatenate(chunks, axis=0)

    @torch.no_grad()
    def encode_texts(self, captions: list[str], batch_size: int = 256) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for start in range(0, len(captions), batch_size):
            batch_caps = captions[start : start + batch_size]
            if self.backend.startswith("transformers:"):
                inputs = self.hf_processor(
                    text=batch_caps,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                feats = self.model.get_text_features(**inputs)
            elif self.backend.startswith("openai_clip:"):
                tokens = self.tokenizer(batch_caps, truncate=True).to(self.device)
                feats = self.model.encode_text(tokens)
            else:
                tokens = self.tokenizer(batch_caps).to(self.device)
                feats = self.model.encode_text(tokens)
            chunks.append(feats.float().cpu().numpy())
        return np.concatenate(chunks, axis=0)


@torch.no_grad()
def extract_image_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Extract unnormalized image embeddings from any PyTorch image encoder."""
    model.eval()
    chunks: list[np.ndarray] = []
    for images, _meta in loader:
        images = images.to(device, non_blocking=True)
        feats = model(images)
        chunks.append(feats.float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _evaluate_and_collect(
    name: str,
    training_type: str,
    dataset_name: str,
    embeddings: np.ndarray,
    compute_whitening: bool = False,
    seed: int = 42,
) -> Tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Compute Table 2 diagnostics, Figure 6 norms, and optional Figs. 9/10 whitening cosines."""
    diag = evaluate_embeddings_both(embeddings)
    norms = compute_norm_stats(embeddings)["norms"].astype(np.float32)

    un = diag["unnormalized"]
    nm = diag["normalized"]
    print(
        f"  {name} ({dataset_name}) | CV={un['cv']:.4f} | "
        f"AD={un['ad_avg']:.3f}/{nm['ad_avg']:.3f} "
        f"({un['ad_norm_feat_pct']:.1f}%/{nm['ad_norm_feat_pct']:.1f}%) | "
        f"DP={un['dp_avg']:.4f}/{nm['dp_avg']:.4f} "
        f"({un['dp_norm_feat_pct']:.1f}%/{nm['dp_norm_feat_pct']:.1f}%)"
    )

    arrays: dict[str, np.ndarray] = {f"{name}_norms": norms}
    whitening_stats: dict[str, Any] = {}

    if compute_whitening:
        # Figure 9: Unnormalized raw vs. ZCA-whitened negative cosine similarities
        z_unnorm_white = zca_whiten(embeddings)
        cos_unnorm_raw = compute_negative_cosines(embeddings, max_pairs=200_000, seed=seed)
        cos_unnorm_white = compute_negative_cosines(z_unnorm_white, max_pairs=200_000, seed=seed)

        # Figure 10: Normalized raw vs. ZCA-whitened negative cosine similarities
        z_norm = l2_normalize_numpy(embeddings)
        z_norm_white = zca_whiten(z_norm)
        cos_norm_raw = compute_negative_cosines(z_norm, max_pairs=200_000, seed=seed)
        cos_norm_white = compute_negative_cosines(z_norm_white, max_pairs=200_000, seed=seed)

        arrays[f"{name}_unnorm_raw_cos"] = cos_unnorm_raw
        arrays[f"{name}_unnorm_white_cos"] = cos_unnorm_white
        arrays[f"{name}_norm_raw_cos"] = cos_norm_raw
        arrays[f"{name}_norm_white_cos"] = cos_norm_white

        whitening_stats = {
            "unnorm_raw_cos_mean": float(np.mean(cos_unnorm_raw)),
            "unnorm_raw_cos_std": float(np.std(cos_unnorm_raw)),
            "unnorm_white_cos_mean": float(np.mean(cos_unnorm_white)),
            "unnorm_white_cos_std": float(np.std(cos_unnorm_white)),
            "norm_raw_cos_mean": float(np.mean(cos_norm_raw)),
            "norm_raw_cos_std": float(np.std(cos_norm_raw)),
            "norm_white_cos_mean": float(np.mean(cos_norm_white)),
            "norm_white_cos_std": float(np.std(cos_norm_white)),
        }

    record = {
        "key": name,
        "training": training_type,
        "dataset": dataset_name,
        "diagnostics": diag,
        "whitening": whitening_stats,
    }
    return record, arrays


def run_pretrained_diagnostics(
    output_dir: str | Path,
    coco_images_dir: str | Path | None = None,
    coco_captions_json: str | Path | None = None,
    imagenet_r_dir: str | Path | None = None,
    models_to_run: list[str] | None = None,
    densenet_variant: str = "densenet121",
    batch_size: int = 128,
    num_workers: int | None = None,
    seed: int = 42,
    device_str: str = "cuda",
) -> dict[str, Any]:
    """Run pretrained model Gaussianity, thin-shell, and whitening evaluations."""
    seed_everything(seed)
    device = get_device(device_str)
    out = ensure_dir(output_dir)
    workers = default_num_workers() if num_workers is None else num_workers
    selected = set(models_to_run or ["resnet34", "densenet", "dino", "clip_image", "clip_text", "clip_imagenet_r"])

    print(f"[Pretrained Diagnostics] Device={device}, models={sorted(selected)}")
    records: list[dict[str, Any]] = []
    saved_arrays: dict[str, np.ndarray] = {}
    pin = torch.cuda.is_available()

    def _make_coco_loader(tf: Callable[[Any], torch.Tensor]) -> Tuple[MSCOCOValDataset, DataLoader]:
        if coco_images_dir is None:
            raise ValueError("Please provide --coco-images-dir pointing to MS-COCO val2017 images.")
        ds = MSCOCOValDataset(
            images_dir=coco_images_dir,
            captions_json=coco_captions_json,
            transform=tf,
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin)
        return ds, loader

    # 1. Supervised ResNet-34 (MS-COCO)
    if "resnet34" in selected and coco_images_dir is not None:
        print("Evaluating Supervised ResNet-34 on MS-COCO...")
        _, loader = _make_coco_loader(_imagenet_eval_transform(224))
        model = ResNet34FeatureExtractor().to(device)
        feats = extract_image_features(model, loader, device)
        rec, arrs = _evaluate_and_collect("resnet34_coco", "Supervised", "MS-COCO", feats, False, seed)
        records.append(rec)
        saved_arrays.update(arrs)
        del model

    # 2. Supervised DenseNet (MS-COCO)
    if "densenet" in selected and coco_images_dir is not None:
        print(f"Evaluating Supervised DenseNet ({densenet_variant}) on MS-COCO...")
        _, loader = _make_coco_loader(_imagenet_eval_transform(224))
        model = DenseNetFeatureExtractor(variant=densenet_variant).to(device)
        feats = extract_image_features(model, loader, device)
        rec, arrs = _evaluate_and_collect("densenet_coco", "Supervised", "MS-COCO", feats, False, seed)
        records.append(rec)
        saved_arrays.update(arrs)
        del model

    # 3. Self-supervised DINO (MS-COCO)
    if "dino" in selected and coco_images_dir is not None:
        print("Evaluating Self-supervised DINO on MS-COCO...")
        dino_model, dino_tf, dino_id = load_dino_model(device)
        print(f"  Loaded DINO backend: {dino_id}")
        _, loader = _make_coco_loader(dino_tf)
        feats = extract_image_features(dino_model, loader, device)
        rec, arrs = _evaluate_and_collect("dino_coco", "Self-supervised", "MS-COCO", feats, True, seed)
        rec["backend"] = dino_id
        records.append(rec)
        saved_arrays.update(arrs)
        del dino_model

    # 4. Self-supervised CLIP ViT-L/14 (Image & Text on MS-COCO, Image on ImageNet-R)
    need_clip = any(m in selected for m in ("clip_image", "clip_text", "clip_imagenet_r"))
    if need_clip and (coco_images_dir is not None or imagenet_r_dir is not None):
        clip_wrapper = CLIPWrapper(device)
        print(f"Loaded CLIP backend: {clip_wrapper.backend}")

        if "clip_image" in selected and coco_images_dir is not None:
            print("Evaluating Self-supervised CLIP (Image) on MS-COCO...")
            coco_ds, loader = _make_coco_loader(clip_wrapper.preprocess)
            feats = clip_wrapper.encode_images(loader)
            rec, arrs = _evaluate_and_collect(
                "clip_image_coco", "Self-supervised", "MS-COCO", feats, True, seed
            )
            rec["backend"] = clip_wrapper.backend
            records.append(rec)
            saved_arrays.update(arrs)

        if "clip_text" in selected and coco_images_dir is not None:
            if coco_captions_json is None:
                raise ValueError("CLIP Text evaluation on MS-COCO requires --coco-captions-json.")
            print("Evaluating Self-supervised CLIP (Text) on MS-COCO captions...")
            coco_ds = MSCOCOValDataset(
                images_dir=coco_images_dir,
                captions_json=coco_captions_json,
                transform=None,
            )
            captions = [cap for _path, cap in coco_ds.samples]
            feats = clip_wrapper.encode_texts(captions, batch_size=batch_size)
            rec, arrs = _evaluate_and_collect(
                "clip_text_coco", "Self-supervised", "MS-COCO", feats, True, seed
            )
            rec["backend"] = clip_wrapper.backend
            records.append(rec)
            saved_arrays.update(arrs)

        if "clip_imagenet_r" in selected and imagenet_r_dir is not None:
            for domain, label in [("sketch", "ImageNet-R Sketch"), ("painting", "ImageNet-R Painting")]:
                print(f"Evaluating Self-supervised CLIP (Image) on {label}...")
                dom_ds = ImageNetRDomainDataset(
                    root_dir=imagenet_r_dir,
                    domain=domain,
                    transform=clip_wrapper.preprocess,
                )
                loader = DataLoader(
                    dom_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin
                )
                feats = clip_wrapper.encode_images(loader)
                rec, arrs = _evaluate_and_collect(
                    f"clip_image_{domain}", "Self-supervised", label, feats, False, seed
                )
                rec["backend"] = clip_wrapper.backend
                records.append(rec)
                saved_arrays.update(arrs)

    summary = {
        "coco_images_dir": str(coco_images_dir) if coco_images_dir else None,
        "coco_captions_json": str(coco_captions_json) if coco_captions_json else None,
        "imagenet_r_dir": str(imagenet_r_dir) if imagenet_r_dir else None,
        "results": records,
    }
    save_json(summary, out / "pretrained_results.json")
    if saved_arrays:
        np.savez_compressed(out / "pretrained_distributions.npz", **saved_arrays)
    print(f"[Pretrained Diagnostics] Saved results to {out}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Gaussianity & whitening on pretrained models.")
    parser.add_argument("--output-dir", type=str, default=str(ROOT_DIR / "results" / "pretrained"))
    parser.add_argument("--coco-images-dir", type=str, default=None, help="Path to MS-COCO val2017 directory.")
    parser.add_argument("--coco-captions-json", type=str, default=None, help="Path to captions_val2017.json.")
    parser.add_argument("--imagenet-r-dir", type=str, default=None, help="Path to ImageNet-R root directory.")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["resnet34", "densenet", "dino", "clip_image", "clip_text", "clip_imagenet_r"],
        choices=["resnet34", "densenet", "dino", "clip_image", "clip_text", "clip_imagenet_r"],
    )
    parser.add_argument("--densenet-variant", type=str, default="densenet121")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pretrained_diagnostics(
        output_dir=args.output_dir,
        coco_images_dir=args.coco_images_dir,
        coco_captions_json=args.coco_captions_json,
        imagenet_r_dir=args.imagenet_r_dir,
        models_to_run=args.models,
        densenet_variant=args.densenet_variant,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        device_str=args.device,
    )

