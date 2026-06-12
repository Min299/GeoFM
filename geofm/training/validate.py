"""geofm.training.validate

Validation utilities for model evaluation.
"""
from __future__ import annotations


def validate(
    runner,
    val_loader,
    task: str = "flood",
):
    """Run validation and return average loss.

    Args:
        runner: EpochRunner instance
        val_loader: Validation data loader
        task: Task name for model forward pass

    Returns:
        Average validation loss
    """
    return runner.validate_epoch(
        val_loader,
        task=task,
    )