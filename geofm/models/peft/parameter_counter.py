"""geofm.models.peft.parameter_counter

Utility functions for counting model parameters.

Helps track PEFT efficiency by measuring:
- Total parameters
- Trainable parameters
- Frozen parameters
- LoRA parameters
- Trainable ratio (PEFT %)
"""
from typing import Dict

import torch.nn as nn

from .lora_layer import LoRALinear


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


def count_lora_params(model: nn.Module) -> int:
    """Count total parameters in LoRA layers.

    Args:
        model: Model with LoRA layers

    Returns:
        Total number of LoRA parameters (lora_A + lora_B)
    """
    total = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            total += module.lora_A.numel() + module.lora_B.numel()
    return total


def count_lora_layers(model: nn.Module) -> int:
    """Count number of LoRA layers.

    Args:
        model: Model with LoRA layers

    Returns:
        Number of LoRA layers
    """
    return sum(1 for module in model.modules() if isinstance(module, LoRALinear))


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


def summary(model: nn.Module) -> Dict[str, any]:
    """Get complete parameter summary.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with all parameter statistics
    """
    total = count_total_params(model)
    trainable = count_trainable_params(model)
    frozen = count_frozen_params(model)
    lora = count_lora_params(model)

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "lora": lora,
        "lora_layers": count_lora_layers(model),
        "trainable_ratio": trainable / total if total > 0 else 0.0,
        "peft_pct": (trainable / total * 100) if total > 0 else 0.0,
    }


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
    stats = summary(model)

    print(f"\n{'=' * 60}")
    print(f"  PARAMETER SUMMARY: {name}")
    print(f"{'=' * 60}")
    print(f"  Total params:     {format_params(stats['total']):>10} ({stats['total']:,})")
    print(f"  Trainable:        {format_params(stats['trainable']):>10} ({stats['trainable']:,})")
    print(f"  Frozen:           {format_params(stats['frozen']):>10} ({stats['frozen']:,})")
    print(f"  LoRA params:      {format_params(stats['lora']):>10} ({stats['lora']:,})")
    print(f"  LoRA layers:      {stats['lora_layers']:>10}")
    print(f"  Trainable ratio:  {stats['peft_pct']:>10.2f}%")
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


def verify_lora_only_trainable(model: nn.Module) -> bool:
    """Verify only LoRA parameters are trainable.

    Args:
        model: Model with LoRA layers injected

    Returns:
        True if only LoRA params are trainable
    """
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            if not param.requires_grad:
                return False
        else:
            if param.requires_grad:
                return False
    return True


class ParameterCounter:
    """Utility class for tracking parameters during training."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.initial = summary(model)

    def get_stats(self) -> Dict[str, any]:
        """Get current parameter statistics.

        Returns:
            Dictionary with parameter stats
        """
        return summary(self.model)

    def print_stats(self):
        """Print current statistics."""
        stats = self.get_stats()
        print(f"  Total: {format_params(stats['total'])} | "
              f"Trainable: {format_params(stats['trainable'])} | "
              f"LoRA: {format_params(stats['lora'])} | "
              f"Ratio: {stats['peft_pct']:.2f}%")