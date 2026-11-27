# PyTorch Reproduction: *InfoNCE Induces Gaussian Distribution* (ICLR 2026)

Complete, self-contained PyTorch implementation of all core experiments from `11492_InfoNCE_Induces_Gaussian-2.pdf` (Betser, Gofer, Levi, & Gilboa, ICLR 2026), derived strictly from the paper text.

## Directory Structure

```text
ai_replication-2/
├── reproduction_plan.md            # Detailed mapping of paper sections, tables, and figures
├── run_all.py                      # Unified CLI orchestrator
├── common/
│   ├── losses.py                   # InfoNCE (Eq. 2), SimCLR NT-Xent, Population Eq. (4) diagnostics
│   ├── metrics.py                  # CV (Eq. 20), AD (< 0.752), DP (p > 0.05), ZCA whitening, pushforward likelihoods
│   ├── models.py                   # LinearEncoder, TwoLayerMLPEncoder, matched ResNet-18 (SimCLR & Supervised)
│   ├── datasets.py                 # Synthetic (Laplace, GMM, Sparse Binary), CIFAR-10, MS-COCO, ImageNet-R
│   └── utils.py                    # Seeding, device selection, JSON I/O
├── experiments/
│   ├── run_synthetic_laplace.py    # Laplace sweep (d in {32..512}, N in {8..128}) & Pushforward (A in {0.2, 0.5, 0.8})
│   ├── run_synthetic_gmm_binary.py # 25-component GMM & Discrete Sparse Binary (E0..E100 every 10 epochs)
│   ├── run_cifar10_mlp.py          # CIFAR-10 2-layer ReLU MLP InfoNCE dynamics (Fig. 4)
│   ├── run_cifar10_resnet18.py     # CIFAR-10 ResNet-18 Contrastive (SimCLR) vs. Supervised (CE) (Table 1)
│   └── run_pretrained_eval.py      # ResNet-34, DenseNet, DINO, CLIP (Image/Text) on MS-COCO & ImageNet-R (Table 2, Figs. 6/9/10)
└── plotting/
    └── generate_figures.py         # Generates Figs. 2, 3, 4, 6, 7, 8, 9, 10, 11 & Tables 1-2 Markdown report
```

## Running the Experiments

All scripts default to `--device cuda` (pass `--device cuda:0` to select a specific GPU, or `--device cpu` / `--device auto` for CPU fallback).

Run individual experiments with your PyTorch + CUDA environment:

```bash
# 1. Synthetic Laplace sweep + Encoder Pushforward (Figs. 2, 3, 7, 8, 11 & Table 1 Laplace)
python experiments/run_synthetic_laplace.py

# 2. Synthetic GMM & Sparse Binary (Table 1 GMM & Binary E0/E50/E100)
python experiments/run_synthetic_gmm_binary.py

# 3. CIFAR-10 Two-Layer MLP Training Dynamics (Fig. 4)
python experiments/run_cifar10_mlp.py

# 4. CIFAR-10 ResNet-18 Contrastive (SimCLR) vs. Supervised (Cross-Entropy) (Table 1)
python experiments/run_cifar10_resnet18.py

# 5. Pretrained Model Diagnostics & Whitening (Table 2, Figs. 6, 9, 10)
python experiments/run_pretrained_eval.py \
    --coco-images-dir /path/to/coco/val2017 \
    --coco-captions-json /path/to/coco/annotations/captions_val2017.json \
    --imagenet-r-dir /path/to/imagenet-r

# 6. Generate all figures and Tables 1 & 2 Markdown summary
python plotting/generate_figures.py
```

Or run the orchestrated pipeline:

```bash
python run_all.py --stages synthetic_laplace synthetic_gmm_binary cifar10_mlp cifar10_resnet18 plot
```

