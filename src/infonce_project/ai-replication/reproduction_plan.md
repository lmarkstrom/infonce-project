# Reproduction Plan: "InfoNCE Induces Gaussian Distribution" (ICLR 2026)

## Paper Summary

The paper proves that InfoNCE-trained representations asymptotically follow a Gaussian distribution. It supports the theory empirically via two routes:
1. **Empirical idealization route**: alignment plateau + thin-shell concentration → uniform on sphere → Gaussian projections
2. **Regularized route**: vanishing KL regularizer → same Gaussian structure

---

## Experiments to Reproduce

### Experiment 1 — Synthetic Laplace Data (Linear Encoder)
**Goal:** Show that InfoNCE on Laplace(0,1) inputs with a linear encoder produces Gaussian representations.

- **Data:** 20k train / 5k test, 1024-dimensional i.i.d. Laplace(0,1) vectors
- **Augmentation:** Two correlated views via `x1 = A*x + sqrt(1-A²)*ε1`, `x2 = A*x + sqrt(1-A²)*ε2`, plus light jitter (additive Gaussian noise std=0.2, feature dropout p=0.1, random multiplicative scaling exp(N(0,0.1²))). A=0.6 default.
- **Encoder:** Linear (1024 → d), representation dims d ∈ {32, 64, 128, 256}
- **Loss:** InfoNCE with temperature τ=0.1
- **Optimizer:** Adam, lr=1e-3, 150 epochs
- **Batch sizes:** N ∈ {8, 16, 32, 48, 64, 96, 128}
- **Metrics:** CV of norms, Anderson-Darling (AD) statistic, D'Agostino-Pearson (DP) p-value, alignment (cosine of positive pairs), uniformity (cosine of negative pairs)
- **Output:** Figures 2 (alignment/uniformity vs batch/dim), Figure 3 (norm stats), Table 1

### Experiment 2 — Synthetic Gaussian Mixture Data (Linear Encoder)
**Goal:** Show Gaussianity emerges regardless of input distribution.

- **Data:** 10k samples from 25-component isotropic Gaussian mixture, 1024-dimensional, random fixed component means
- **Augmentation:** Positive pairs = two independent samples from the same mixture component
- **Encoder:** Linear (1024 → 256)
- **Loss:** InfoNCE, τ=0.1
- **Optimizer:** Adam, lr=1e-3, 100 epochs
- **Metrics:** CV, AD, DP
- **Output:** Table 1

### Experiment 3 — Synthetic Binary Data (Linear Encoder)
**Goal:** Show Gaussianity emerges from discrete binary inputs.

- **Data:** 10k sparse binary vectors of dimension 1024
- **Augmentation:** Positive pairs by independently flipping 0.1% of zero entries to ones
- **Encoder:** Linear (1024 → 256)
- **Loss:** InfoNCE, τ=0.1
- **Optimizer:** Adam, lr=1e-3, 100 epochs
- **Evaluation:** Every 10 epochs (Table 1 shows Binary E0/E50/E100)
- **Metrics:** CV, AD, DP
- **Output:** Table 1

### Experiment 4 — CIFAR-10, Two-Layer MLP Contrastive (Training Dynamics)
**Goal:** Show Gaussianity increases over training epochs.

- **Data:** CIFAR-10 (50k train, 10k test)
- **Augmentations:** SimCLR-style: random resized crop to 32×32 (scale 0.2–1.0), random horizontal flip, color jitter (0.8, 0.8, 0.8, 0.2), random grayscale p=0.2
- **Encoder:** Two-layer MLP with single ReLU nonlinearity, input 3072 (flattened 32×32×3), hidden and output dimensions to 256
- **Loss:** InfoNCE, τ=0.1
- **Optimizer:** Adam, lr=1e-3, weight_decay=1e-4, batch_size=256, 100 epochs
- **Metrics:** CV, AD statistic, DP fraction over epochs
- **Output:** Figure 4 (training dynamics)

### Experiment 5 — CIFAR-10, ResNet-18, Contrastive vs. Supervised
**Goal:** Show that Gaussianity is due to the InfoNCE objective, not architecture/data.

- **Data:** CIFAR-10 (50k train, 10k test)
- **Augmentations:** SimCLR-style (same as above) for contrastive; standard for supervised
- **Encoder:** ResNet-18 (following SimCLR protocol)
- **Contrastive:** InfoNCE, τ=0.1, Adam, lr=1e-3, weight_decay=1e-4, batch_size=256, 100 epochs
- **Supervised:** Cross-entropy, same optimizer and batch size
- **Metrics:** CV, AD (avg + fraction), DP (avg + fraction)
- **Output:** Table 1

---

## Metrics Implementation

### Coefficient of Variation (CV)
```
CV = std({||z_i||}) / mean({||z_i||})
```
A small CV (≈0.08-0.09 for contrastive) indicates thin-shell concentration.

### Anderson-Darling (AD) Test
- Per coordinate, test statistic < 0.752 → accept normality
- Report average statistic and fraction of coordinates passing

### D'Agostino-Pearson (DP) Test
- Per coordinate, p-value > 0.05 → accept normality  
- Report average p-value and fraction of coordinates passing

### Alignment
- Mean cosine similarity of positive pairs: `mean(u_i · v_i)`

### Uniformity
- Mean cosine similarity of negative pairs: `mean_{i≠j}(u_i · v_j)`

---

## File Structure

```
ai-replication/
├── reproduction_plan.md          (this file)
├── utils.py                      (metrics: CV, AD, DP, alignment, uniformity)
├── infonce_loss.py               (InfoNCE loss implementation)
├── models.py                     (LinearEncoder, MLP, ResNet-18 wrapper)
├── augmentations.py              (synthetic augmentations, SimCLR augmentations)
├── datasets.py                   (synthetic Laplace, GMM, Binary datasets)
├── train_synthetic.py            (Experiments 1, 2, 3 — synthetic data)
├── train_cifar_mlp.py            (Experiment 4 — CIFAR-10 MLP dynamics)
├── train_cifar_resnet.py         (Experiment 5 — CIFAR-10 ResNet contrastive vs supervised)
└── evaluate.py                   (Shared evaluation and plotting utilities)
```

---

## Execution Checklist

- [x] InfoNCE loss implementation
- [x] CV metric
- [x] AD test metric
- [x] DP test metric
- [x] Alignment/uniformity metrics
- [x] Synthetic Laplace data + augmentations
- [x] Gaussian mixture data + augmentations
- [x] Binary data + augmentations
- [x] Linear encoder
- [x] Two-layer MLP encoder
- [x] ResNet-18 encoder
- [x] SimCLR CIFAR-10 augmentations
- [x] Synthetic data training loop (Exp 1, 2, 3) → Table 1
- [x] CIFAR-10 MLP training with dynamics tracking → Figure 4
- [x] CIFAR-10 ResNet-18 contrastive vs supervised → Table 1
- [x] Plots: Figure 2 (alignment/uniformity vs batch/dim)
- [x] Plots: Figure 3 (norm stats)
- [x] Plots: Figure 4 (training dynamics)
- [x] Table 1 reproduction
