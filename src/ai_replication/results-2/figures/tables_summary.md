# Reproduced Diagnostics Tables (Tables 1 & 2)

## Table 1: Gaussianity Diagnostics Across Data and Training Settings (Unnormalized)

| Metric | Laplace | GMM | Binary E0 | Binary E50 | Binary E100 | CIFAR-10 Supervised | CIFAR-10 Contrastive |
|---|---:|---:|---:|---:|---:|---:|---:|
| CV | 0.054 | 0.028 | 0.079 | 0.076 | 0.077 | 0.275 | 0.124 |
| AD Avg. (< 0.752) | 0.378 | 0.391 | 0.383 | 0.398 | 0.404 | 1157.755 | 1362.806 |
| AD Norm. Feat. | 95.3% | 94.9% | 92.6% | 93.8% | 93.0% | 0.0% | 0.0% |
| DP Avg. (> 0.05) | 0.513 | 0.456 | 0.465 | 0.458 | 0.456 | 0.000 | 0.000 |
| DP Norm. Feat. | 95.3% | 94.1% | 94.9% | 93.0% | 91.4% | 0.0% | 0.0% |
| Gaussian? | Yes | Yes | Yes | Yes | Yes | No | No |

## Table 2: Gaussianity Diagnostics for Pretrained Models (Unnormalized / Normalized)

_Pretrained diagnostics have not been executed yet._
