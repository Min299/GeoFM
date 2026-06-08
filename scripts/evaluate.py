#!/usr/bin/env python3
"""scripts/evaluate.py

Evaluation script for GeoFM models.

Usage:
    python scripts/evaluate.py --task flood --checkpoint outputs/model.pt
    python scripts/evaluate.py --task burn --checkpoint outputs/burn_model.pt
"""
import argparse
import sys
from pathlib import Path

import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from geofm.experiments import ExperimentConfig
from geofm.experiments.experiment_builder import ExperimentBuilder
from geofm.evaluation import SegmentationMetrics, get_metrics
from geofm.trainers import CheckpointManager


def create_dummy_dataloader(batch_size: int = 8):
    """Create a dummy dataloader for testing.

    In production, replace with actual dataset.
    """
    from torch.utils.data import DataLoader, TensorDataset

    images = torch.randn(batch_size, 12, 224, 224)
    targets = torch.randint(0, 2, (batch_size, 224, 224))

    dataset = TensorDataset(images, targets)
    return DataLoader(dataset, batch_size=batch_size)


def main():
    parser = argparse.ArgumentParser(description="Evaluate GeoFM model")

    parser.add_argument(
        "--task",
        required=True,
        choices=["flood", "burn", "lulc", "ndvi"],
        help="Task type",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--adaptation",
        type=str,
        default="lora",
        choices=["feature", "lora", "hybrid", "full_ft"],
        help="Adaptation method",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.csv",
        help="Output CSV for results",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Adaptation: {args.adaptation}")

    # Create config
    config = ExperimentConfig(
        name=f"eval_{args.task}",
        task=args.task,
        adaptation=args.adaptation,
        backbone="terramind_base",
        batch_size=args.batch_size,
    )

    # Build model
    print("Building model...")
    model = ExperimentBuilder.build(config)

    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint_manager = CheckpointManager()
        checkpoint_manager.load(args.checkpoint, model)
        print("Checkpoint loaded successfully")

    model.eval()

    # Get metrics for task
    task_metrics = get_metrics(args.task)
    print(f"Metrics: {task_metrics}")

    # Create dataloader
    dataloader = create_dummy_dataloader(args.batch_size)

    # Initialize metrics tracker
    seg_metrics = SegmentationMetrics()

    # Run evaluation
    print("Running evaluation...")
    with torch.no_grad():
        for i, (images, targets) in enumerate(dataloader):
            # Dummy forward pass - in production, this would be the actual model
            logits = torch.randn(args.batch_size, 2, 224, 224)

            seg_metrics.update(logits, targets)

            if i >= 2:  # Limit to 3 batches for demo
                break

    # Compute results
    results = seg_metrics.compute()

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    for key, value in results.items():
        if key in ["tp", "fp", "fn", "tn"]:
            continue
        print(f"{key}: {value:.4f}")

    # Save results
    print(f"\nResults would be saved to: {args.output}")

    print("\nEvaluation completed")


if __name__ == "__main__":
    main()