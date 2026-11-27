"""Master CLI orchestrator for reproducing all experiments in 'InfoNCE Induces Gaussian Distribution'.

Runs any combination of:
- synthetic_laplace  (Figs. 2, 3, 7, 8, 11 & Table 1 Laplace)
- synthetic_gmm_binary (Table 1 GMM & Binary E0/E50/E100)
- cifar10_mlp        (Fig. 4)
- cifar10_resnet18   (Table 1 CIFAR-10 ResNet-18 Supervised vs. Contrastive)
- pretrained         (Table 2, Figs. 6, 9, 10)
- plot               (Generates all figures & Tables 1-2 Markdown report)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.run_cifar10_mlp import run_cifar10_mlp_experiment
from experiments.run_cifar10_resnet18 import run_cifar10_resnet18_experiment
from experiments.run_pretrained_eval import run_pretrained_diagnostics
from experiments.run_synthetic_gmm_binary import run_additional_synthetic_experiments
from experiments.run_synthetic_laplace import run_laplace_experiments
from plotting.generate_figures import generate_all_outputs

ALL_STAGES = [
    "synthetic_laplace",
    "synthetic_gmm_binary",
    "cifar10_mlp",
    "cifar10_resnet18",
    "pretrained",
    "plot",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full PyTorch reproduction of 'InfoNCE Induces Gaussian Distribution'."
    )
    parser.add_argument(
        "--stages",
        type=str,
        nargs="+",
        default=["synthetic_laplace", "synthetic_gmm_binary", "cifar10_mlp", "cifar10_resnet18", "plot"],
        choices=ALL_STAGES,
        help="Experiment stages to execute in order.",
    )
    parser.add_argument("--results-dir", type=str, default=str(ROOT_DIR / "results"))
    parser.add_argument("--data-dir", type=str, default=str(ROOT_DIR / "data" / "cifar10"))
    parser.add_argument("--coco-images-dir", type=str, default=None)
    parser.add_argument("--coco-captions-json", type=str, default=None)
    parser.add_argument("--imagenet-r-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-pretrained-resnet18", action="store_true")
    parser.add_argument("--cifar-stem", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_dir)

    if "synthetic_laplace" in args.stages:
        run_laplace_experiments(
            output_dir=results_root / "synthetic_laplace",
            seed=args.seed,
            device_str=args.device,
        )

    if "synthetic_gmm_binary" in args.stages:
        run_additional_synthetic_experiments(
            output_dir=results_root / "synthetic_gmm_binary",
            seed=args.seed,
            device_str=args.device,
        )

    if "cifar10_mlp" in args.stages:
        run_cifar10_mlp_experiment(
            data_dir=args.data_dir,
            output_dir=results_root / "cifar10_mlp",
            seed=args.seed,
            device_str=args.device,
        )

    if "cifar10_resnet18" in args.stages:
        run_cifar10_resnet18_experiment(
            data_dir=args.data_dir,
            output_dir=results_root / "cifar10_resnet18",
            pretrained=not args.no_pretrained_resnet18,
            cifar_stem=args.cifar_stem,
            seed=args.seed,
            device_str=args.device,
        )

    if "pretrained" in args.stages:
        run_pretrained_diagnostics(
            output_dir=results_root / "pretrained",
            coco_images_dir=args.coco_images_dir,
            coco_captions_json=args.coco_captions_json,
            imagenet_r_dir=args.imagenet_r_dir,
            seed=args.seed,
            device_str=args.device,
        )

    if "plot" in args.stages:
        generate_all_outputs(
            results_root=results_root,
            figures_dir=results_root / "figures",
        )


if __name__ == "__main__":
    main()

