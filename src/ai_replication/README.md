## Agent Isolation & Jail Configuration

**Model:** Cloude Sonnet4.5

To satisfy the course requirement of reproducing the machine learning paper **solely from the text provided** and preventing the LLM from accessing external code or repositories, the following strict isolation measures were enforced in Google Antigravity:

- **Technical Network Jail:**
  - Configured a global wildcard deny rule (`*`) for `read_url` permissions to block all external web browsing and URL scraping.
  - Restricted terminal and command execution to manual review to prevent unauthorized CLI downloads (`curl`, `wget`, `git clone`).
- **File System Scope Isolation:**
  - Removed workspace and file access permissions to any reference implementations or existing codebases outside the isolated assignment directory.

## System prompt

In the ai-replication folder, do this:

Read the assignment and rules in our GEMINI.md file. Following those constraints strictly, write a complete PyTorch implementation of all main experiments for the research paper in the ai-replication folder using only the pdf paper.

---

## Reproduction Guide: Step-by-Step

### 1. Environment & Setup

From the root of the project or within this directory:

```bash
cd /Users/linus/Desktop/school/infonce-project/src/infonce_project/ai-replication
# Use the workspace venv python:
VENV_PY="../../../.venv/bin/python"
```

### 2. Execution Options

#### Option A: Run All Experiments

To run all experiments sequentially (Synthetic, CIFAR-10 MLP, CIFAR-10 ResNet):

```bash
# Fast smoke-test (few epochs, smaller batch/dim grids):
$VENV_PY run_all.py --quick

# Full paper-fidelity run:
$VENV_PY run_all.py
```

#### Option B: Run Experiments Individually

1. **Synthetic Experiments (Laplace, GMM, Binary)**
   - Reproduces: **Figure 2**, **Figure 3**, and **Table 1 (Synthetic columns)**

   ```bash
   $VENV_PY train_synthetic.py            # Full run (150 epochs)
   $VENV_PY train_synthetic.py --quick    # Quick run (30 epochs)
   ```

2. **CIFAR-10 Training Dynamics (2-Layer MLP)**
   - Reproduces: **Figure 4 (Norm concentration, AD statistic, DP % over epochs)**

   ```bash
   $VENV_PY train_cifar_mlp.py --epochs 100
   $VENV_PY train_cifar_mlp.py --quick
   ```

3. **CIFAR-10 ResNet-18 (Contrastive vs. Supervised)**
   - Reproduces: **Table 1 (ResNet-18 Supervised vs. Contrastive columns)**

   ```bash
   $VENV_PY train_cifar_resnet.py --epochs 100
   $VENV_PY train_cifar_resnet.py --quick
   ```

4. **Generate Consolidated Report**
   ```bash
   $VENV_PY evaluate.py
   ```

---

## How to Retrieve Metrics & Compare with Paper

All metrics evaluate unnormalized representations $z = f(x) \in \mathbb{R}^d$ on held-out test data.

### 1. The Core Metrics (Defined in `utils.py`)

- **Norm Concentration (CV - Eq. 20):**
  $$\text{CV} = \frac{\text{std}(\{\|z_i\|\}_{i=1}^N)}{\text{mean}(\{\|z_i\|\}_{i=1}^N)}$$
  Lower CV indicates stronger thin-shell concentration on the hypersphere.
- **Anderson-Darling (AD) Test:**
  Evaluated per-coordinate. Critical value at $\alpha=0.05$ is $0.752$. Coordinates with statistic $< 0.752$ pass the normality hypothesis. Reported as **AD Avg.** and **AD Norm. Feat. (%)**.
- **D'Agostino-Pearson (DP) Test:**
  Evaluated per-coordinate. Coordinates with $p > 0.05$ fail to reject normality. Reported as **DP Avg.** and **DP Norm. Feat. (%)**.
- **Alignment:** Mean cosine similarity of positive pairs $\mathbb{E}_{(u,v) \sim \pi}[u \cdot v]$.
- **Uniformity:** Mean cosine similarity of negative pairs $\mathbb{E}_{i \neq j}[u_i \cdot u_j]$.

### 2. Output Artifacts in `results/`

Generated files map directly to figures and tables in the paper:

| Generated Output                        | Paper Counterpart         | Description                                             |
| :-------------------------------------- | :------------------------ | :------------------------------------------------------ |
| `results/synthetic_table1.txt`          | **Table 1** (Columns 1–5) | Laplace, GMM, and Binary (E0, E50, E100) diagnostics    |
| `results/cifar10_resnet_table1.txt`     | **Table 1** (Columns 6–7) | ResNet-18 Supervised vs. Contrastive diagnostics        |
| `results/fig2_alignment_uniformity.png` | **Figure 2**              | Alignment and uniformity vs. batch size & dimension     |
| `results/fig3_norm_stats.png`           | **Figure 3**              | CV and AD stats vs. batch size curves across dimensions |
| `results/fig4_cifar10_mlp_dynamics.png` | **Figure 4**              | Evolution of CV, AD, and DP % over training epochs      |
| `results/full_report.txt`               | Consolidated summary      | Side-by-side metric dump across all completed runs      |

### 3. Paper Reference Baselines (Table 1 Benchmark)

| Setting                              |  CV  | AD Avg (<0.752) | AD Norm % | DP Avg (>0.05) | DP Norm % | Gaussian? |
| :----------------------------------- | :--: | :-------------: | :-------: | :------------: | :-------: | :-------: |
| **Laplace (Linear)**                 | 0.08 |      0.38       |   100%    |      0.49      |   100%    |   **✓**   |
| **GMM (Linear)**                     | 0.08 |      0.39       |   100%    |      0.46      |   100%    |   **✓**   |
| **Binary E0**                        | 0.36 |      1.64       |    30%    |      0.02      |    15%    |   **✗**   |
| **Binary E50**                       | 0.12 |      0.41       |    93%    |      0.44      |    89%    |   **✓**   |
| **Binary E100**                      | 0.09 |      0.42       |    97%    |      0.46      |    98%    |   **✓**   |
| **CIFAR-10 Supervised (ResNet-18)**  | 0.50 |      3.30       |   6.2%    |     0.041      |   3.9%    |   **✗**   |
| **CIFAR-10 Contrastive (ResNet-18)** | 0.09 |      0.43       |   96.1%   |     0.390      |   94.5%   |   **✓**   |

---

## Reproduction Guide: Step-by-Step

### 1. Environment & Setup

From the root of the project or within this directory:

```bash
cd /Users/linus/Desktop/school/infonce-project/src/infonce_project/ai-replication
# Use the workspace venv python:
VENV_PY="../../../.venv/bin/python"
```

### 2. Execution Options

#### Option A: Run All Experiments

To run all experiments sequentially (Synthetic, CIFAR-10 MLP, CIFAR-10 ResNet):

```bash
# Fast smoke-test (few epochs, smaller batch/dim grids):
$VENV_PY run_all.py --quick

# Full paper-fidelity run:
$VENV_PY run_all.py
```

#### Option B: Run Experiments Individually

1. **Synthetic Experiments (Laplace, GMM, Binary)**
   - Reproduces: **Figure 2**, **Figure 3**, and **Table 1 (Synthetic columns)**

   ```bash
   $VENV_PY train_synthetic.py            # Full run (150 epochs)
   $VENV_PY train_synthetic.py --quick    # Quick run (30 epochs)
   ```

2. **CIFAR-10 Training Dynamics (2-Layer MLP)**
   - Reproduces: **Figure 4 (Norm concentration, AD statistic, DP % over epochs)**

   ```bash
   $VENV_PY train_cifar_mlp.py --epochs 100
   $VENV_PY train_cifar_mlp.py --quick
   ```

3. **CIFAR-10 ResNet-18 (Contrastive vs. Supervised)**
   - Reproduces: **Table 1 (ResNet-18 Supervised vs. Contrastive columns)**

   ```bash
   $VENV_PY train_cifar_resnet.py --epochs 100
   $VENV_PY train_cifar_resnet.py --quick
   ```

4. **Generate Consolidated Report**
   ```bash
   $VENV_PY evaluate.py
   ```

---

## How to Retrieve Metrics & Compare with Paper

All metrics evaluate unnormalized representations $z = f(x) \in \mathbb{R}^d$ on held-out test data.

### 1. The Core Metrics (Defined in `utils.py`)

- **Norm Concentration (CV - Eq. 20):**
  $$\text{CV} = \frac{\text{std}(\{\|z_i\|\}_{i=1}^N)}{\text{mean}(\{\|z_i\|\}_{i=1}^N)}$$
  Lower CV indicates stronger thin-shell concentration on the hypersphere.
- **Anderson-Darling (AD) Test:**
  Evaluated per-coordinate. Critical value at $\alpha=0.05$ is $0.752$. Coordinates with statistic $< 0.752$ pass the normality hypothesis. Reported as **AD Avg.** and **AD Norm. Feat. (%)**.
- **D'Agostino-Pearson (DP) Test:**
  Evaluated per-coordinate. Coordinates with $p > 0.05$ fail to reject normality. Reported as **DP Avg.** and **DP Norm. Feat. (%)**.
- **Alignment:** Mean cosine similarity of positive pairs $\mathbb{E}_{(u,v) \sim \pi}[u \cdot v]$.
- **Uniformity:** Mean cosine similarity of negative pairs $\mathbb{E}_{i \neq j}[u_i \cdot u_j]$.

### 2. Output Artifacts in `results/`

Generated files map directly to figures and tables in the paper:

| Generated Output                        | Paper Counterpart         | Description                                             |
| :-------------------------------------- | :------------------------ | :------------------------------------------------------ |
| `results/synthetic_table1.txt`          | **Table 1** (Columns 1–5) | Laplace, GMM, and Binary (E0, E50, E100) diagnostics    |
| `results/cifar10_resnet_table1.txt`     | **Table 1** (Columns 6–7) | ResNet-18 Supervised vs. Contrastive diagnostics        |
| `results/fig2_alignment_uniformity.png` | **Figure 2**              | Alignment and uniformity vs. batch size & dimension     |
| `results/fig3_norm_stats.png`           | **Figure 3**              | CV and AD stats vs. batch size curves across dimensions |
| `results/fig4_cifar10_mlp_dynamics.png` | **Figure 4**              | Evolution of CV, AD, and DP % over training epochs      |
| `results/full_report.txt`               | Consolidated summary      | Side-by-side metric dump across all completed runs      |

### 3. Paper Reference Baselines (Table 1 Benchmark)

| Setting                              |  CV  | AD Avg (<0.752) | AD Norm % | DP Avg (>0.05) | DP Norm % | Gaussian? |
| :----------------------------------- | :--: | :-------------: | :-------: | :------------: | :-------: | :-------: |
| **Laplace (Linear)**                 | 0.08 |      0.38       |   100%    |      0.49      |   100%    |   **✓**   |
| **GMM (Linear)**                     | 0.08 |      0.39       |   100%    |      0.46      |   100%    |   **✓**   |
| **Binary E0**                        | 0.36 |      1.64       |    30%    |      0.02      |    15%    |   **✗**   |
| **Binary E50**                       | 0.12 |      0.41       |    93%    |      0.44      |    89%    |   **✓**   |
| **Binary E100**                      | 0.09 |      0.42       |    97%    |      0.46      |    98%    |   **✓**   |
| **CIFAR-10 Supervised (ResNet-18)**  | 0.50 |      3.30       |   6.2%    |     0.041      |   3.9%    |   **✗**   |
| **CIFAR-10 Contrastive (ResNet-18)** | 0.09 |      0.43       |   96.1%   |     0.390      |   94.5%   |   **✓**   |
