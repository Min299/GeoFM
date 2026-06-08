"""geofm.debug.parameter_report

Parameter counting and reporting utilities.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """Count total parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Total number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_frozen_parameters(model: nn.Module) -> int:
    """Count frozen (non-trainable) parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Number of frozen parameters
    """
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)


def parameter_report(model: nn.Module) -> Dict[str, int]:
    """Generate a parameter report for a model.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with total, trainable, and frozen parameter counts
    """
    total = count_parameters(model)
    trainable = count_trainable_parameters(model)
    frozen = count_frozen_parameters(model)

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
    }


def detailed_parameter_report(
    model: nn.Module,
    module_name: Optional[str] = None,
) -> Dict:
    """Generate a detailed parameter report.

    Args:
        model: PyTorch model
        module_name: Optional name for the report

    Returns:
        Dictionary with detailed parameter breakdown
    """
    report = {
        "total": 0,
        "trainable": 0,
        "frozen": 0,
        "by_module": {},
    }

    for name, module in model.named_modules():
        module_params = sum(p.numel() for p in module.parameters())
        module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)

        if module_params > 0:
            report["by_module"][name] = {
                "total": module_params,
                "trainable": module_trainable,
                "frozen": module_params - module_trainable,
            }

        report["total"] += module_params
        report["trainable"] += module_trainable

    report["frozen"] = report["total"] - report["trainable"]

    return report


def format_params(num_params: int) -> str:
    """Format parameter count in human-readable form.

    Args:
        num_params: Number of parameters

    Returns:
        Formatted string (e.g., "1.2M" or "500K")
    """
    if num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.1f}M"
    elif num_params >= 1_000:
        return f"{num_params / 1_000:.1f}K"
    else:
        return str(num_params)


def print_parameter_report(
    model: nn.Module,
    name: str = "Model",
) -> None:
    """Print a formatted parameter report.

    Args:
        model: PyTorch model
        name: Name to display in report
    """
    report = parameter_report(model)

    print("\n" + "=" * 60)
    print(f"Parameter Report: {name}")
    print("=" * 60)
    print(f"Total Parameters:     {report['total']:,} ({format_params(report['total'])})")
    print(f"Trainable Parameters: {report['trainable']:,} ({format_params(report['trainable'])})")
    print(f"Frozen Parameters:    {report['frozen']:,} ({format_params(report['frozen'])})")

    if report['total'] > 0:
        trainable_pct = 100 * report['trainable'] / report['total']
        print(f"Trainable Percentage: {trainable_pct:.2f}%")

    print("=" * 60)


def peft_ratio(model: nn.Module) -> float:
    """Calculate PEFT (Parameter-Efficient Fine-Tuning) ratio.

    Args:
        model: PyTorch model

    Returns:
        Percentage of trainable parameters
    """
    total = count_parameters(model)
    if total == 0:
        return 0.0
    return 100 * count_trainable_parameters(model) / total