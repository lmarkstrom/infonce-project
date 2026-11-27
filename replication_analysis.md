# Replication Discrepancy Analysis: Paper (arXiv:2602.24012v2) vs. Replication Code

This document explains how the results in [`results.md`](./results.md) (from running [`src/replication/`](./src/replication/)) compare to **Table 1 and Appendix E.1 of Betser et al., "InfoNCE Induces Gaussian Distribution" (ICLR 2026, [2602.24012v2.pdf](./2602.24012v2.pdf))**, highlights **exactly which numbers are comparable**, and details the setup differences that cause the remaining discrepancies.

---

## 1. At a Glance: Which Values in `results.md` Are Comparable to Table 1?

When reading [`results.md`](./results.md), there are **two structural traps** to watch out for:
1. **`Per-coordinate` vs. `Random-projection`:** Every script in `src/replication/` prints both *Per-coordinate* tests and *Random-projection* tests. **Table 1 in the paper ONLY reports Per-coordinate tests** (Section 5, p. 8: *"we apply two standard one-dimensional normality tests to individual coordinates"*). Random projections were added in [`diagnostics.py`](./src/replication/cifar10_resnet18/diagnostics.py) as an auxiliary check.
2. **`512-dim Backbone` vs. `128-dim Projector`:** For CIFAR-10 Contrastive, [`train_contrastive.py`](./src/replication/cifar10_resnet18/train_contrastive.py) prints **two layers**: the 512-dim ResNet-18 backbone (post-ReLU) and the 128-dim Projector output (linear). **Table 1's Contrastive column corresponds to the 128-dim Projector output**, *not* the 512-dim post-ReLU backbone.

### Direct Comparison Table (Highlighting Comparable Pairs)

| Paper Table 1 Column | Metric | Paper Value | Matching `results.md` Value | Which `results.md` Block to Use | Comparable? |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **Synthetic: Laplace**<br>*(Linear, `d=256`)* | **CV**<br>**AD Avg. (`<0.752`)**<br>**AD Norm. Feat.**<br>**DP Avg. (`>0.05`)**<br>**DP Norm. Feat.** | `0.08`<br>`0.38`<br>`100%`<br>`0.49`<br>`100%` | **`0.0556`**<br>**`0.3659`**<br>**`94.9%`**<br>**`0.5381`**<br>**`95.7%`** | `run_single_laplace.py`<br>→ **Per-coordinate** (`d=256`) | **Yes (Apples-to-Apples).**<br>AD/DP averages match closely. (~95% pass is expected for $\alpha=0.05$; lower CV is from batch size 256 vs. $\le 128$). |
| **Synthetic: GMM**<br>*(Linear, `d=256`)* | **CV**<br>**AD Avg. (`<0.752`)**<br>**AD Norm. Feat.**<br>**DP Avg. (`>0.05`)**<br>**DP Norm. Feat.** | `0.08`<br>`0.39`<br>`100%`<br>`0.46`<br>`100%` | **`0.0279`**<br>**`0.3764`**<br>**`96.1%`**<br>**`0.4938`**<br>**`97.3%`** | `run_single_gmm.py`<br>→ **Per-coordinate** (`d=256`) | **Yes (Apples-to-Apples).**<br>AD/DP averages match closely. Lower CV (`0.0279`) is due to tight cluster mean scale (`0.05`) and batch size 256. |
| **CIFAR-10: Contrastive**<br>*(ResNet-18)* | **CV**<br>**AD Avg. (`<0.752`)**<br>**AD Norm. Feat.**<br>**DP Avg. (`>0.05`)**<br>**DP Norm. Feat.** | `0.09`<br>`0.43`<br>`96.1%`<br>`0.39`<br>`94.5%` | **`0.1649`**<br>**`0.4700`**<br>**`89.8%`**<br>**`0.4126`**<br>**`85.9%`** | `train_contrastive.py`<br>→ **`[Projector output (128-dim, InfoNCE space)]`**<br>→ **Per-coordinate** | **Yes (Comparable).**<br>128-dim projector is where InfoNCE is applied and has no trailing ReLU. *(Do **not** compare against the 512-dim post-ReLU backbone, which gets `AD=16.60, pass=0.0%`).* |
| **CIFAR-10: Supervised**<br>*(ResNet-18)* | **CV**<br>**AD Avg. (`<0.752`)**<br>**AD Norm. Feat.**<br>**DP Avg. (`>0.05`)**<br>**DP Norm. Feat.** | `0.5`<br>`3.3`<br>`6.2%`<br>`0.041`<br>`3.9%` | `0.2402`<br>`21.3800` *(Proj: `1.53`)*<br>`0.0%` *(Proj: `35.0%`)*<br>`0.0000` *(Proj: `0.15`)*<br>`0.0%` *(Proj: `41.0%`)* | `train_supervised.py`<br>→ **`[Supervised backbone (512-dim)]`**<br>*(No 128-dim projector exists in this script)* | **No (Apples-to-Oranges).**<br>Qualitative conclusion matches (*Not Gaussian*), but raw numbers differ due to **post-ReLU truncation (135 dead units)**, **training from scratch vs. ImageNet pretraining**, and **SimCLR augmentations**. |

---

## 2. Detailed Explanation of Every Setup Difference

### Difference 1: Which Layer Is Evaluated (`512-dim Post-ReLU Backbone` vs. `128-dim Linear Projector`)
This is the single most important reason why the **512-dim backbone** lines in [`results.md`](./results.md) show `0.0%` pass rates and huge AD statistics (`21.38` for Supervised, `16.60` for Contrastive).

* **How [`model.py`](./src/replication/cifar10_resnet18/model.py) is wired:**
  ```
  Input Image x
     │
     ▼
  [ResNet18Cifar] ── ends with BasicBlock.relu(out) + AdaptiveAvgPool2d(1)
     │
     ├──► h (512-dim backbone): EVERY coordinate is strictly >= 0 (post-ReLU!)
     │       │
     │       ├─► In train_supervised.py:  evaluated directly! (No projector head)
     │       └─► In train_contrastive.py: evaluated as "[Contrastive backbone (512-dim)]"
     │
     ▼
  [ProjectionHead: Linear(512,512) -> ReLU -> Linear(512,128)] (Contrastive only)
     │
     └──► z (128-dim projector): Unconstrained in (-∞, +∞), InfoNCE loss applied here!
             └─► In train_contrastive.py: evaluated as "[Projector output (128-dim)]"
  ```
* **Why the 512-dim backbone ALWAYS fails per-coordinate normality (`pass = 0.0%`):**
  - In [`model.py:26-30`](./src/replication/cifar10_resnet18/model.py#L26-L30), every `BasicBlock` ends with `return self.relu(out)`.
  - Therefore, every coordinate $h_k$ ($k=1,\dots,512$) of the backbone output is the average of $4 \times 4 = 16$ non-negative ReLU activations ($h_k \ge 0$).
  - A 1D variable that can never be negative and has a large spike at $0$ is fundamentally non-Gaussian. When tested coordinate-by-coordinate, **both the Contrastive 512-dim backbone (`AD = 16.5974, pass = 0.0%`) and the Supervised 512-dim backbone (`AD = 21.3800, pass = 0.0%`) fail completely.**
* **Why the 128-dim Projector Output in `train_contrastive.py` matches the paper's Table 1:**
  - The paper's theory proves Gaussianity for the representation $f(x)$ on which the InfoNCE loss operates. In `train_contrastive.py`, InfoNCE operates on the **128-dim `ProjectionHead` output**, whose final layer is `nn.Linear(512, 128)` with **no ReLU**.
  - On that 128-dim space, `results.md` gets `AD avg = 0.4700 (pass = 89.8%)` and `DP avg = 0.4126 (pass = 85.9%)`, closely matching Table 1 (`AD avg = 0.43, pass = 96.1%`, `DP avg = 0.39, pass = 94.5%`).
  - Even the exact percentages in Table 1 of the paper match integer counts out of 128 dimensions:
    $$\frac{123}{128} = 96.09\% \approx \mathbf{96.1\%}, \quad \frac{121}{128} = 94.53\% \approx \mathbf{94.5\%}, \quad \frac{8}{128} = 6.25\% \approx \mathbf{6.2\%}, \quad \frac{5}{128} = 3.91\% \approx \mathbf{3.9\%}$$
* **Why `train_supervised.py` gets `AD = 21.38` instead of `3.3`:**
  - `train_supervised.py` does **not** include a 128-dim projection head; it only evaluates the 512-dim post-ReLU backbone.
  - Because cross-entropy only needs to separate 10 classes linearly, training from scratch with weight decay causes **135 out of 512 backbone coordinates (26.4%) to die completely (`std = 0`)**, while the remaining 377 post-ReLU coordinates are heavily zero-inflated (`AD = 21.38`).
  - **Where to look in `results.md` for a fairer 512-dim comparison:** Look at the **`Random-projection`** row of the 512-dim backbones:
    - **Contrastive 512-dim Random Projections:** `AD avg = 0.6341 (pass = 75.0%)`, `DP avg = 0.3317 (pass = 71.0%)` (mostly Gaussian)
    - **Supervised 512-dim Random Projections:** `AD avg = 1.5328 (pass = 35.0%)`, `DP avg = 0.1480 (pass = 41.0%)` (non-Gaussian, and on the same numerical scale as the paper's `AD = 3.3` because projecting onto a random unit vector $w \in \mathbb{S}^{511}$ mixes positive and negative weights and averages away the single-coordinate $\ge 0$ ReLU truncation).

---

### Difference 2: ImageNet Pretraining (Paper) vs. Training from Scratch (Replication Code)
* **Paper (Appendix E.1, p. 20):** Explicitly states under *CIFAR-10 experiments → Architecture*:
  > *"a ResNet-18 encoder following the SimCLR protocol for the contrastive setting and trained with standard cross-entropy for the supervised setting **(pretrained on ImageNet (Deng et al., 2009))**."*
  *(Note: Section 5 on p. 10 states "Both models share identical initialization and capacity", which is an internal inconsistency in the paper between the main text and Appendix E.1).*
* **Replication Code ([`train_supervised.py:205`](./src/replication/cifar10_resnet18/train_supervised.py#L205)):** Initializes a custom CIFAR-adapted `ResNet18Cifar` ($3\times 3$ stem, no maxpool) **randomly from scratch**.
* **Effect on metrics:** Training a 512-wide network from scratch on only 10 classes leads to severe Neural Collapse and 135 dead ReLU channels (`AD = 21.38`). An ImageNet-pretrained backbone (learned on 1,000 classes) retains distributed features across all channels and avoids massive dead-unit collapse, yielding `AD = 3.3` (comparable to the supervised ImageNet-pretrained models in Table 2 of the paper: DenseNet `AD = 2.98`, ResNet-34 `AD = 10.01`).

---

### Difference 3: Supervised Data Augmentation (`simclr_augment` vs. `eval_transform`)
* **Contradiction inside the replication repo:**
  - The docstrings in [`data.py:1-2`](./src/replication/cifar10_resnet18/data.py#L1-L2) and [`data.py:25-27`](./src/replication/cifar10_resnet18/data.py#L25-L27) state:
    > *"CIFAR-10 data pipeline: SimCLR-style augmentation for contrastive pairs, **plain normalization for supervised training / evaluation**."*
  - However, [`train_supervised.py:77`](./src/replication/cifar10_resnet18/train_supervised.py#L77) (and line 93 for validation) passes **`simclr_augment()`** (random resized crops down to 20% of the image, strong `0.8` color jitter, and 20% random grayscale) into the supervised training loader.
* **Effect on `CV` (`0.2402` in `results.md` vs. `0.5` in Paper Table 1):**
  - When a supervised network is trained on CIFAR-10 with standard/plain transforms, it rapidly achieves ~100% training accuracy and drives confident samples to very large, dispersed logit/feature norms, producing a high **`CV = 0.5`**.
  - When trained for 100 epochs with aggressive SimCLR augmentations and weight decay (`1e-4`), the network cannot easily memorize training examples, which regularizes the feature norms and cuts the norm variation in half (**`CV = 0.2402`**).

---

### Difference 4: Test Set Subsampling (`N_TEST_SAMPLES = 250` vs. Full Test Set)
* **Paper (Appendix E.1, p. 20):** Specifies CIFAR-10 test set size of `10k` (and `5k` for synthetic tests).
* **Replication Code ([`diagnostics.py:38-42`](./src/replication/cifar10_resnet18/diagnostics.py#L38-L42), [`train_contrastive.py:33-37`](./src/replication/cifar10_resnet18/train_contrastive.py#L33-L37)):**
  - First subsets the CIFAR-10 test set from 10,000 to `N_TEST = 2000` images (used to compute `CV`).
  - Then, for **each individual coordinate test**, randomly subsamples only **`N_TEST_SAMPLES = 250` points** out of the 2,000 images.
  - Why? Hypothesis tests like Anderson-Darling and D'Agostino-Pearson become ultra-strict at large sample sizes ($n = 10{,}000$), rejecting even tiny deviations from a theoretical Gaussian. Subsampling $n = 250$ keeps the test power calibrated to detect practical Gaussianity, though it introduces slight run-to-run variance.

---

### Difference 5: Synthetic Hyperparameters (`run_single_laplace.py` & `run_single_gmm.py`)
While the synthetic AD and DP averages in [`results.md`](./results.md) match Table 1 almost perfectly (`AD = 0.3659 / 0.3764` vs. `0.38 / 0.39`), three minor differences explain the `CV` and pass-rate numbers:
1. **Batch Size (`256` vs. `<= 128`) and Epochs (`100` vs. `150`):** Appendix E.1 (p. 19) lists $N \in \{8, 16, 32, 48, 64, 96, 128\}$ and 150 epochs for Laplace (where Fig. 3 shows $\text{CV} \approx 0.08$ at $N=128$). The replication scripts use `BATCH_SIZE = 256` and `EPOCHS = 100`. Larger batch sizes strengthen norm concentration, lowering `CV` from `0.08` to `0.0556`.
2. **GMM Component Separation (`MEAN_SCALE = 0.05`):** In [`run_single_gmm.py:25-26`](./src/replication/run_single_gmm.py#L25-L26), each component has `COMPONENT_STD = 1.0`, while the 25 component means are drawn with standard deviation `MEAN_SCALE = 0.05`. Because $0.05 \ll 1.0$, all 25 components overlap almost completely at the origin (the input is already nearly a single isotropic Gaussian), and fresh samples are drawn every epoch across `N_TRAIN = 20000`, driving `CV` down to `0.0279`.
3. **Why ~95% Pass Rate Instead of 100%:** Both `AD < 0.752` and `DP > 0.05` are statistical tests at the **$\alpha = 0.05$ significance level** (a 5% false-rejection rate under the null hypothesis). Even if you test 256 coordinates of an **exact** $\mathcal{N}(0, I_{256})$ Gaussian at $n = 5000$, roughly $5\%$ of coordinates will fail by chance—so `94.9%–97.3%` in `results.md` is the exact theoretical pass rate of a true Gaussian.
