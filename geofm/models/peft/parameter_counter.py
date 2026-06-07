"""geofm.models.peft.parameter_counter

Utility functions for counting model parameters.

Helps track PEFT efficiency by measuring:
- Total parameters
- Trainable parameters
- Frozen parameters
- Trainable ratio (PEFT %)
"""
from typing import Dict

import torch.nn as nn


def count_total_params(model: nn.Module) -> int:
    """Count total parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Total parameter count
    """
    return sum(p.numel() for p in model.parameters())


def count_trainable_params(model: nn.Module) -> int:
    """Count trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_frozen_params(model: nn.Module) -> int:
    """Count frozen parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of frozen parameters
    """
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)


def trainable_ratio(model: nn.Module) -> float:
    """Calculate trainable parameter ratio.

    Args:
        model: PyTorch model

    Returns:
        Ratio of trainable to total parameters (0.0 to 1.0)
    """
    total = count_total_params(model)
    if total == 0:
        return 0.0
    return count_trainable_params(model) / total


def peft_percentage(model: nn.Module) -> float:
    """Calculate PEFT percentage.

    Same as trainable_ratio but returns as percentage.

    Args:
        model: PyTorch model

    Returns:
        Percentage of trainable parameters (0.0 to 100.0)
    """
    return trainable_ratio(model) * 100.0


def count_params_by_component(model: nn.Module) -> Dict[str, int]:
    """Count parameters grouped by component name.

    Args:
        model: PyTorch model

    Returns:
        Dictionary mapping component prefixes to param counts
    """
    counts = {}

    for name, param in model.named_parameters():
        # Extract the top-level component name
        parts = name.split('.')
        component = parts[0] if parts else "unknown"

        if component not in counts:
            counts[component] = 0

        counts[component] += param.numel()

    return counts


def format_params(num: int) -> str:
    """Format parameter count for display.

    Args:
        num: Parameter count

    Returns:
        Formatted string (e.g., "1.2M", "85.5M")
    """
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return str(num)


def print_param_summary(model: nn.Module, name: str = "Model"):
    """Print a formatted parameter summary.

    Args:
        model: PyTorch model
        name: Name to display
    """
    total = count_total_params(model)
    trainable = count_trainable_params(model)
    frozen = count_frozen_params(model)
    ratio = trainable_ratio(model)

    print(f"\n{'=' * 60}")
    print(f"  PARAMETER SUMMARY: {name}")
    print(f"{'=' * 60}")
    print(f"  Total params:     {format_params(total):>10} ({total:,})")
    print(f"  Trainable:        {format_params(trainable):>10} ({trainable:,})")
    print(f"  Frozen:           {format_params(frozen):>10} ({frozen:,})")
    print(f"  Trainable ratio:  {ratio * 100:>10.2f}%")
    print(f"{'=' * 60}\n")


def verify_peft_ready(model: nn.Module, max_ratio: float = 0.05) -> bool:
    """Verify a model is PEFT-ready (trainable ratio below threshold).

    Args:
        model: PyTorch model
        max_ratio: Maximum allowed trainable ratio (default: 5%)

    Returns:
        True if model is PEFT-ready
    """
    ratio = trainable_ratio(model)

    if ratio > max_ratio:
        print(f"WARNING: Trainable ratio {ratio * 100:.2f}% exceeds {max_ratio * 100}% threshold")
        return False

    return True


class ParameterCounter:
    """Utility class for tracking parameters during training."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.initial_total = count_total_params(model)
        self.initial_trainable = count_trainable_params(model)

    def get_stats(self) -> Dict[str, any]:
        """Get current parameter statistics.

        Returns:
            Dictionary with parameter stats
        """
        return {
            "total": count_total_params(self.model),
            "trainable": count_trainable_params(self.model),
            "frozen": count_frozen_params(self.model),
            "ratio": trainable_ratio(self.model),
            "peft_pct": peft_percentage(self.model),
        }

    def print_stats(self):
        """Print current statistics."""
        stats = self.get_stats()
        print(f"  Total: {format_params(stats['total'])} | "
              f"Trainable: {format_params(stats['trainable'])} | "
              f"Ratio: {stats['peft_pct']:.2f}%")