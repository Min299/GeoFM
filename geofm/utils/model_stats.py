"""geofm.utils.model_stats

Model statistics utilities.
"""
from __future__ import annotations

import torch


def count_parameters(model: torch.nn.Module) -> int:
    """Count total number of parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Total number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_frozen_parameters(model: torch.nn.Module) -> int:
    """Count frozen (non-trainable) parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of frozen parameters
    """
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)


def get_model_summary(model: torch.nn.Module) -> dict:
    """Get comprehensive model statistics.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with model statistics
    """
    total = count_parameters(model)
    trainable = count_trainable_parameters(model)
    frozen = count_frozen_parameters(model)

    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
        "trainable_percentage": 100 * trainable / total if total > 0 else 0,
    }


def print_model_summary(model: torch.nn.Module, name: str = "Model") -> None:
    """Print formatted model summary.

    Args:
        model: PyTorch model
        name: Model name for display
    """
    summary = get_model_summary(model)

    print("=" * 60)
    print(name.upper())
    print("=" * 60)
    print(f"Total Parameters:     {summary['total_parameters']:>12,}")
    print(f"Trainable Parameters: {summary['trainable_parameters']:>12,} ({summary['trainable_percentage']:.2f}%)")
    print(f"Frozen Parameters:    {summary['frozen_parameters']:>12,}")
    print("=" * 60)