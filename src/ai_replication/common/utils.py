"""Utility helpers for reproducibility, device management, and result persistence."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(requested: str | None = "cuda") -> torch.device:
    """Select computation device, targeting CUDA by default.

    - 'cuda' / 'cuda:N': uses CUDA and raises a clear error if CUDA is unavailable.
    - 'auto' / None: uses CUDA when available, otherwise falls back to CPU.
    - 'cpu': forces CPU execution.
    """
    if requested is None or requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device '{requested}', but CUDA is not available on this machine. "
            "Run on a CUDA-enabled machine or pass '--device cpu'."
        )
    return torch.device(requested)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (including parents) if it does not exist."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _to_json_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if np.isnan(val) or np.isinf(val) else val
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return None if (obj != obj or obj in (float("inf"), float("-inf"))) else obj
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(data: Any, path: str | Path) -> Path:
    """Write a Python dictionary/list to JSON with NumPy type handling."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(_to_json_serializable(data), f, indent=2)
    return target


def load_json(path: str | Path) -> Any:
    """Read a JSON file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def default_num_workers() -> int:
    """Choose a safe default DataLoader worker count for the host OS."""
    cpu_count = os.cpu_count() or 2
    return min(4, max(0, cpu_count // 2))

