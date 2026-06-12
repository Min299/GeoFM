"""geofm.utils.seed

Reproducibility utilities for experiments.
"""
from __future__ import annotations

import random
import os

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms (slower but reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Also set deterministic flag for PyTorch operations
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_seed() -> int:
    """Get current random seed.

    Returns:
        Current seed or -1 if not set
    """
    return torch.initial_seed() if torch.cuda.is_initialized() or hasattr(torch, 'initial_seed') else -1


def seed_worker(worker_id: int) -> None:
    """Worker function for DataLoader seeding.

    Usage:
        DataLoader(..., worker_init_fn=seed_worker)
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)