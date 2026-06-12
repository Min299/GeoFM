#!/usr/bin/env python3
"""scripts/inspect_model.py

Inspect GeoFM model architecture and parameters.

Usage:
    python scripts/inspect_model.py
    python scripts/inspect_model.py --adaptation lora --task flood
"""
import argparse
import sys
from pathlib import Path

import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from geofm.experiments import ExperimentConfig
from geofm.experiments.experiment_builder import ExperimentBuilder


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return total, trainable, frozen


def inspect(model, name: str = "Model"):
    """Inspect model and print statistics."""
    total, trainable, frozen = count_parameters(model)

    print("=" * 60)
    print(name.upper())
    print("=" * 60)
    print()
    print(model)
    print()
    print("-" * 60)
    print("PARAMETER STATISTICS")
    print("-" * 60)
    print(f"Total Parameters:     {total:>12,}")
    print(f"Trainable Parameters: {trainable:>12,} ({100*trainable/total:.2f}%)")
    print(f"Frozen Parameters:    {frozen:>12,} ({100*frozen/total:.2f}%)")
    print()

    # Print layer breakdown
    print("-" * 60)
    print("LAYER BREAKDOWN")
    print("-" * 60)

    layer_stats = {}
    for name, param in model.named_parameters():
        layer_type = name.split(".")[-1] if "." in name else name
        if layer_type not in layer_stats:
            layer_stats[layer_type] = {"count": 0, "params": 0}
        layer_stats[layer_type]["count"] += 1
        layer_stats[layer_type]["params"] += param.numel()

    for layer_type, stats in sorted(layer_stats.items()):
        print(f"  {layer_type:20s}: {stats['count']:3d} layers, {stats['params']:>10,} params")

    print()


def main():
    parser = argparse.ArgumentParser(description="Inspect GeoFM model")

    parser.add_argument(
        "--task",
        type=str,
        default="flood",
        choices=["flood", "burn", "lulc"],
        help="Task type",
    )
    parser.add_argument(
        "--adaptation",
        type=str,
        default="lora",
        choices=["feature", "lora", "hybrid", "full_ft"],
        help="Adaptation method",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="terramind_base",
        help="Backbone model",
    )

    args = parser.parse_args()

    print("Building model for inspection...")

    # Create config
    config = ExperimentConfig(
        name=f"inspect_{args.adaptation}",
        task=args.task,
        adaptation=args.adaptation,
        backbone=args.backbone,
    )

    # Build model
    model = ExperimentBuilder.build(config)

    if model is None:
        print("Error: Model build returned None")
        print("This adaptation type may not be fully implemented yet.")
        return

    # Inspect
    inspect(model, f"GeoFM ({args.adaptation})")

    # Print CUDA info
    print("-" * 60)
    print("DEVICE INFO")
    print("-" * 60)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


if __name__ == "__main__":
    main()