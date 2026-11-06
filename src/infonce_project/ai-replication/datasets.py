"""
Datasets for the synthetic experiments described in Section 5 and Appendix E.1.

Three synthetic settings:
  1. Laplace(0,1) — independent coordinates, ddata=1024
  2. Gaussian mixture — 25 components, 1024-dimensional, random means
  3. Binary sparse — 1024-dimensional 0/1 vectors

All datasets expose a __getitem__ that returns (x, label) where
label is the component index (used for positive-pair generation in GMM).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class LaplaceDataset(Dataset):
    """
    i.i.d. Laplace(0, 1) vectors.

    Paper spec (Appendix E.1):
      - ddata = 1024
      - 20k train, 5k test

    Args:
        n_samples: Number of samples.
        dim: Dimension of each sample. Default 1024.
        seed: Random seed for reproducibility.
    """

    def __init__(self, n_samples: int = 20_000, dim: int = 1024, seed: int = 42):
        rng = np.random.default_rng(seed)
        # Laplace(0,1): loc=0, scale=1
        data = rng.laplace(loc=0.0, scale=1.0, size=(n_samples, dim)).astype(np.float32)
        self.data = torch.from_numpy(data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], 0  # label not used for Laplace


class GaussianMixtureDataset(Dataset):
    """
    Gaussian mixture with 25 equally weighted components.

    Paper spec (Appendix E.1):
      - 25 components, 1024 dimensions
      - Shared isotropic covariance (identity)
      - Random means fixed at initialization
      - 10k samples

    Args:
        n_samples: Number of samples. Default 10000.
        dim: Dimension. Default 1024.
        n_components: Number of Gaussian components. Default 25.
        seed: Random seed.
    """

    def __init__(
        self,
        n_samples: int = 10_000,
        dim: int = 1024,
        n_components: int = 25,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)

        # Fixed random component means
        means = rng.standard_normal((n_components, dim)).astype(np.float32)

        # Assign each sample to a component (uniform weights)
        component_ids = rng.integers(0, n_components, size=n_samples)

        # Sample from selected components
        data = np.empty((n_samples, dim), dtype=np.float32)
        for i in range(n_samples):
            c = component_ids[i]
            data[i] = means[c] + rng.standard_normal(dim)

        self.data = torch.from_numpy(data)
        self.component_ids = torch.from_numpy(component_ids.astype(np.int64))
        self.means = torch.from_numpy(means)
        self.n_components = n_components

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], self.component_ids[idx]


class BinaryDataset(Dataset):
    """
    Sparse binary vectors of dimension 1024.

    Paper spec (Appendix E.1):
      - 1024-dimensional sparse binary vectors
      - 10k samples

    Implementation: each sample has a small fraction of 1s;
    we draw a Bernoulli with p=0.05 to get a reasonable sparsity
    while remaining within the spirit of "sparse binary".

    Args:
        n_samples: Number of samples. Default 10000.
        dim: Dimension. Default 1024.
        sparsity: Probability of a 1-entry. Default 0.05.
        seed: Random seed.
    """

    def __init__(
        self,
        n_samples: int = 10_000,
        dim: int = 1024,
        sparsity: float = 0.05,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        data = (rng.random((n_samples, dim)) < sparsity).astype(np.float32)
        self.data = torch.from_numpy(data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], 0  # label not used
