# Reproduction Plan: "InfoNCE Induces Gaussian Distribution" (ICLR 2026)

This plan maps every core experiment, dataset, architecture, objective, hyperparameter, diagnostic metric, table, and figure described in `11492_InfoNCE_Induces_Gaussian-2.pdf` (Betser, Gofer, Levi, & Gilboa, ICLR 2026) into a complete, self-contained PyTorch implementation.

---

## 1. Core Formulation & Shared Infrastructure

### 1.1 Objectives (`common/losses.py`)
1. **Empirical InfoNCE Loss (Section 3, Eq. 2)**:
   - Given $\ell_2$-normalized representations $u_i = \hat{f}(x_i)$, $v_i = \hat{f}(y_i) \in \mathbb{S}^{d-1}$ (with deterministic fallback $c_0 = e_1 \in \mathbb{S}^{d-1}$ when $\|f(x)\|_2 = 0$) and fixed temperature $\tau > 0$:
     $$\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{N}\sum_{i=1}^N \log \frac{\exp(\frac{1}{\tau}\langle u_i, v_i \rangle)}{\sum_{j=1}^N \exp(\frac{1}{\tau}\langle u_i, v_j \rangle)}.$$
   - Also supports symmetric InfoNCE and SimCLR NT-Xent ($2N$ batch with self-masking) for the SimCLR ResNet-18 protocol.
2. **Supervised Baseline Loss**: Standard `nn.CrossEntropyLoss` for CIFAR-10 ResNet-18 comparison.
3. **Population Alignment & Uniformity Diagnostics (Section 3, Eq. 4)**:
   - Alignment: $\mathbb{E}_{(u,v)\sim\pi}[u \cdot v]$.
   - Uniformity potential: $\Phi(\mu) = \mathbb{E}_{u\sim\mu}\log \mathbb{E}_{v\sim\mu}\exp(\frac{1}{\tau} u \cdot v)$.

### 1.2 Diagnostic Metrics (`common/metrics.py`)
All experiments evaluate **both unnormalized** ($z = f(x)$) and **normalized** ($u = \hat{f}(x)$) test-set representations:
1. **Norm Concentration (Thin-Shell Metric, Section 5, Eq. 20)**:
   - Coefficient of Variation: $\text{CV} = \text{std}(\{\|z_i\|_2\}_{i=1}^N) / \text{mean}(\{\|z_i\|_2\}_{i=1}^N)$, plus mean and std of $\|z_i\|_2$.
2. **Coordinate-Wise Normality Tests (Section 5)**:
   - **Anderson-Darling (AD)** test (`scipy.stats.anderson(..., dist="norm")`):
     - `AD Avg.`: mean AD statistic across all $d$ coordinates.
     - `AD Norm. Feat. (%)`: percentage of coordinates with $\text{AD} < 0.752$.
   - **D'Agostino-Pearson (DP)** test (`scipy.stats.normaltest`):
     - `DP Avg.`: mean $p$-value across all $d$ coordinates.
     - `DP Norm. Feat. (%)`: percentage of coordinates with $p > 0.05$.
3. **Cosine Similarity Distributions (Alignment vs. Uniformity)**:
   - Positive-pair cosine similarities $\langle u_i, v_i \rangle$ and negative-pair cosine similarities $\langle u_i, v_j \rangle$ ($i \neq j$).
4. **Post-Hoc ZCA Whitening (Appendix E.2, Figs. 9–10)**:
   - Center representations, estimate sample covariance $\Sigma$, apply $Z_{\text{white}} = (Z - \bar{Z}) \Sigma^{-1/2}$, and compare negative cosine similarity distributions before and after whitening.
5. **Encoder Pushforward Log-Likelihood Correlation (Appendix E.2, Fig. 11)**:
   - Source exact Laplace(0, 1) log-likelihood $\log p_{\text{data}}(x) = \sum_{j=1}^{d_{\text{data}}}(-\log 2 - |x_j|)$ vs. fitted multivariate Gaussian log-likelihood $\log \mathcal{N}(z; \hat{\mu}, \hat{\Sigma})$ on test representations, quantified by Pearson $r$.

---

## 2. Main Experiments Mapping

### Experiment 1: Synthetic Laplace Data & Pushforward Analysis (`experiments/run_synthetic_laplace.py`)
- **Paper Targets**: Figure 2, Figure 3, Figure 7, Figure 8, Figure 11, Table 1 (`Synthetic Linear - Laplace`).
- **Dataset**:
  - i.i.d. $\text{Laplace}(0, 1)$ vectors with $d_{\text{data}} = 1024$.
  - Split: `20,000` training samples, `5,000` test samples (Appendix E.1).
- **Augmentations (Eq. 53 & Appendix E.1)**:
  - Correlated Gaussian mixing: $x_1 = A x + \sqrt{1 - A^2}\varepsilon_1$, $x_2 = A x + \sqrt{1 - A^2}\varepsilon_2$, $\varepsilon_1, \varepsilon_2 \sim \mathcal{N}(0, I)$, with default $A = 0.6$.
  - Independent jitter per view: additive Gaussian noise ($\sigma = 0.2$), feature dropout ($p = 0.1$), random multiplicative scaling by $\exp(\mathcal{N}(0, 0.1^2))$.
- **Model**: Linear encoder `nn.Linear(1024, d)`.
- **Hyperparameters**:
  - Sweep dimensions $d \in \{32, 64, 128, 256, 512\}$ (covering Appendix E.1 and Figs. 2, 3, 7, 8).
  - Sweep batch sizes $N \in \{8, 16, 32, 48, 64, 96, 128\}$.
  - Temperature $\tau = 0.1$ (configurable to $0.2$).
  - Optimizer: Adam, $\text{lr} = 10^{-3}$, `150` epochs (configurable 50–250).
- **Pushforward Sub-Experiment (Fig. 11)**:
  - Train linear encoders across augmentation strengths $A \in \{0.2, 0.5, 0.8\}$ and compute Pearson correlation between $\log p_{\text{Laplace}}(x)$ and $\log p_{\mathcal{N}(\hat{\mu}, \hat{\Sigma})}(z)$ on test samples.

### Experiment 2: Additional Synthetic Distributions — GMM & Discrete Sparse Binary (`experiments/run_synthetic_gmm_binary.py`)
- **Paper Targets**: Table 1 (`Synthetic Linear - GMM`, `Synthetic Linear - Binary E0/E50/E100`).
- **2a. Gaussian Mixture Model (GMM)**:
  - `10,000` training samples (plus held-out test set from the same fixed components) in $\mathbb{R}^{1024}$ from 25 equally weighted Gaussian components with independently sampled fixed means and shared isotropic covariance.
  - Augmentation: positive pairs drawn independently from the same underlying mixture component.
  - Model & Optimization: `nn.Linear(1024, 256)`, InfoNCE ($\tau = 0.1$), Adam ($\text{lr} = 10^{-3}$), `100` epochs.
- **2b. Discrete Sparse Binary**:
  - `10,000` training samples (plus held-out test set) of sparse binary vectors in $\{0, 1\}^{1024}$.
  - Augmentation: positive pairs generated by independently flipping `0.1%` of zero entries to ones.
  - Model & Optimization: `nn.Linear(1024, 256)`, InfoNCE ($\tau = 0.1$), Adam ($\text{lr} = 10^{-3}$), `100` epochs.
  - Evaluation schedule: evaluate CV, AD, and DP diagnostics every `10` epochs (including E0, E50, E100).

### Experiment 3: CIFAR-10 Two-Layer MLP Training Dynamics (`experiments/run_cifar10_mlp.py`)
- **Paper Targets**: Figure 4 (CV, AD statistic, and DP passing feature fraction over training epochs).
- **Dataset**: CIFAR-10 (`50,000` train, `10,000` test).
- **Augmentations (Appendix E.1)**: SimCLR pipeline — `RandomResizedCrop(32, scale=(0.2, 1.0))`, `RandomHorizontalFlip`, `ColorJitter(0.8, 0.8, 0.8, 0.2)` (p=0.8), `RandomGrayscale(p=0.2)`.
- **Model**: Two-layer MLP with a single ReLU nonlinearity (`3072 -> hidden_dim -> ReLU -> repr_dim`, default `512 -> 256`).
- **Optimization**: InfoNCE ($\tau = 0.1$), Adam, $\text{lr} = 10^{-3}$, weight decay $= 10^{-4}$, batch size $= 256$, `100` epochs.
- **Evaluation**: Compute CV, AD, and DP metrics on the CIFAR-10 test set at epoch 0 and throughout training (every epoch or configured interval).

### Experiment 4: CIFAR-10 ResNet-18 Contrastive vs. Supervised (`experiments/run_cifar10_resnet18.py`)
- **Paper Targets**: Table 1 (`CIFAR-10 ResNet-18 - Supervised` vs. `Contrastive`).
- **Dataset**: CIFAR-10 (`50,000` train, `10,000` test), SimCLR augmentation pipeline for contrastive views.
- **Architecture & Identical Initialization**:
  - Shared ResNet-18 backbone (ImageNet-pretrained by default per Appendix E.1, with exact initial state dict cloned to both branches so initialization and capacity are strictly identical).
  - **Contrastive branch (SimCLR)**: 512-dim ResNet-18 backbone + 2-layer MLP projection head (`512 -> 512 -> ReLU -> 128`), trained with InfoNCE / NT-Xent ($\tau = 0.1$).
  - **Supervised branch**: identical 512-dim ResNet-18 backbone + linear classifier (`512 -> 10`), trained with cross-entropy loss.
- **Optimization**: Adam, $\text{lr} = 10^{-3}$, weight decay $= 10^{-4}$, batch size $= 256$, `100` epochs.
- **Evaluation**:
  - Extract 512-dim penultimate representations on CIFAR-10 test set (both unnormalized and normalized) and compute CV, AD Avg / Norm. Feat. %, DP Avg / Norm. Feat. %.
  - Also report supervised test accuracy and contrastive linear-probe accuracy to verify representation quality.

### Experiment 5: Pretrained Foundation & Supervised Model Diagnostics (`experiments/run_pretrained_eval.py`)
- **Paper Targets**: Table 2, Figure 6, Figure 9, Figure 10.
- **Models**:
  - Supervised (ImageNet-pretrained): `ResNet-34`, `DenseNet-121` (`torchvision.models`).
  - Self-supervised: `DINO (ViT-B/32)` (with `timm`/`transformers`/`torch.hub` loader) and `CLIP (ViT-L/14)` image & text encoders (`open_clip`/`clip`/`transformers` loader).
- **Datasets**:
  - Full MS-COCO 2017 validation set (`5,000` images + captions).
  - Full ImageNet-R benchmark (`Sketch` and `Painting` domains for CLIP Image).
- **Diagnostics**:
  - Unnormalized and normalized coordinate-wise AD & DP statistics (Table 2).
  - Unnormalized representation norm histograms and CV (Figure 6).
  - Raw vs. ZCA-whitened negative cosine similarity histograms for unnormalized (Figure 9) and normalized (Figure 10) embeddings of CLIP (Image), CLIP (Text), and DINO.

---

## 3. Visualization, Reporting & Orchestration

- `plotting/generate_figures.py`: Generates high-resolution figures reproducing Figures 2, 3, 4, 6, 7, 8, 9, 10, and 11 from saved experiment JSON/NPZ outputs, plus formatted Markdown & LaTeX tables reproducing Tables 1 and 2.
- `run_all.py`: Unified command-line entry point to execute individual experiments or the complete reproduction pipeline end-to-end.

---

## 4. Verification Plan
1. Unit-test and smoke-test all modules (losses, metrics, datasets, models, training loops, whitening, pushforward correlation, figure/table generation) using the project's virtual environment.
2. Verify all synthetic experiments end-to-end and confirm expected qualitative and quantitative trends (thin-shell CV decay, AD < 0.752, DP p > 0.05, alignment saturation vs. uniformity improvement, binary E0 non-Gaussian -> E50/E100 Gaussian transition, pushforward Pearson correlation).
3. Verify CIFAR-10 MLP & ResNet-18 (Contrastive vs. Supervised) pipelines and pretrained model evaluation scripts execute cleanly with real data loaders, no placeholders, and complete logging.

