"""geofm.training.eval_step

Evaluation step implementation.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def eval_step(
    model: torch.nn.Module,
    batch: dict,
    criterion: torch.nn.Module,
    task: str = "flood",
) -> float:
    """Execute one evaluation step.

    Args:
        model: The model
        batch: Batch dictionary with 'inputs' and 'mask'
        criterion: Loss function
        task: Task name

    Returns:
        Loss value
    """
    model.eval()

    logits = model(batch["inputs"], task_name=task)
    loss = criterion(logits, batch["mask"])

    return loss.item()


@torch.no_grad()
def eval_step_with_metrics(
    model: torch.nn.Module,
    batch: dict,
    criterion: torch.nn.Module,
    metrics: dict,
    task: str = "flood",
) -> dict:
    """Execute evaluation step with metrics.

    Args:
        model: The model
        batch: Batch dictionary
        criterion: Loss function
        metrics: Dictionary of metric functions
        task: Task name

    Returns:
        Dictionary with loss and metrics
    """
    model.eval()

    logits = model(batch["inputs"], task_name=task)
    loss = criterion(logits, batch["mask"])

    results = {"loss": loss.item()}

    # Compute metrics
    preds = logits.argmax(dim=1)
    targets = batch["mask"]

    for name, metric_fn in metrics.items():
        results[name] = metric_fn(preds, targets)

    return results