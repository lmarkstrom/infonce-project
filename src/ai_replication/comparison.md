# Implementation Comparison: `ai-replication` vs. `replication` (Original)

This document provides a concise architectural and functional comparison between our clean paper-only reproduction (`ai-replication/`) and the original author repository (`replication/`).

---

## 1. High-Level Summary

| Dimension | `replication/` (Original Repo) | `ai-replication/` (Our Reproduction) |
| :--- | :--- | :--- |
| **Completeness** | Implements Laplace, GMM, and CIFAR ResNet-18. **Omits** Binary Sparse and MLP dynamics. | **Complete**: Implements all 5 main experiments (Laplace grid, GMM, Binary E0/E50/E100, MLP dynamics, ResNet-18). |
| **Code Structure** | Standalone scripts (`run_single_laplace.py`, `run_single_gmm.py`, `cifar10_resnet18/`). | Modularized pipeline (`models.py`, `datasets.py`, `augmentations.py`, `utils.py`, `run_all.py`). |
| **Loss Function** | Symmetric InfoNCE on $L_2$-normalized vectors with $\tau = 0.1$. | **Identical**: Symmetric InfoNCE on $L_2$-normalized vectors with $\tau = 0.1$. |
| **Synthetic Laplace Augmentation** | Exact Eq. 53 ($A=0.6$, noise std 0.2, dropout 0.1, log-normal scale 0.1). | **Identical**: Exact Eq. 53 formulation. |
| **CIFAR ResNet Architecture** | Custom CIFAR-adapted ResNet-18 (3×3 stem, no maxpool). | `torchvision.models.resnet18` backbone with SimCLR projection head. |
| **Gaussianity Diagnostics** | Coordinate-wise AD/DP + random 1D projections; subsamples 250 points. | Full coordinate-wise AD/DP across test set; computes exact Table 1 metrics. |
| **Automation & Execution** | Manual per-script execution. | Single unified runner `run_all.py` with fast smoke-test `--quick` flag. |

---

## 2. Key Differences in Detail

### A. Experiment Coverage
- **Original (`replication/`)**:
  - Focused only on three experiments: Laplace, GMM, and ResNet-18 CIFAR-10.
  - Did not implement the discrete **Sparse Binary** setting (Table 1: E0/E50/E100) or the **Two-Layer MLP Training Dynamics** (Figure 4).
- **Our Implementation (`ai-replication/`)**:
  - Implements the complete set of main experiments specified in the paper, strictly fulfilling the reproduction constraints.

### B. CIFAR-10 ResNet-18 Stem
- **Original**: Uses a custom-written `ResNet18Cifar` where the first layer is a $3 \times 3$ convolution with `stride=1` and no `maxpool`. This is the standard adaptation for $32 \times 32$ CIFAR images.
- **Our Implementation**: Uses PyTorch's native `torchvision.models.resnet18`, replacing `fc` with a SimCLR 2-layer MLP projection head (`512 -> 512 -> 256`).

### C. GMM Component Scaling
- **Original**: Scales component means by `MEAN_SCALE = 0.05` so the 25 cluster centers stay close relative to the component variance.
- **Our Implementation**: Sampled raw standard normal means $\mathcal{N}(0, I)$, making cluster separation larger prior to training.

### D. Diagnostics & Sampling
- **Original**: Evaluates AD and DP by repeatedly drawing subsets of 250 samples (`N_TEST_SAMPLES = 250`) and also tests random 1D projections $\mathbf{w}^T z$.
- **Our Implementation**: Evaluates the diagnostics across all test samples per coordinate directly, perfectly mirroring the column metrics presented in Table 1.
