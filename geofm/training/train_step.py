"""geofm.training.train_step

Training step implementation.
"""
from __future__ import annotations

import torch


def train_step(
    model: torch.nn.Module,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    task: str = "flood",
) -> float:
    """Execute one training step.

    Args:
        model: The model
        batch: Batch dictionary with 'inputs' and 'mask'
        optimizer: Optimizer
        criterion: Loss function
        task: Task name (flood, burn, lulc)

    Returns:
        Loss value
    """
    model.train()

    optimizer.zero_grad()

    # Forward pass
    logits = model(batch["inputs"], task_name=task)

    # Compute loss
    loss = criterion(logits, batch["mask"])

    # Backward pass
    loss.backward()

    # Optimizer step
    optimizer.step()

    return loss.item()


def train_step_with_grad_clip(
    model: torch.nn.Module,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    max_grad_norm: float = 1.0,
    task: str = "flood",
) -> float:
    """Execute one training step with gradient clipping.

    Args:
        model: The model
        batch: Batch dictionary
        optimizer: Optimizer
        criterion: Loss function
        max_grad_norm: Maximum gradient norm
        task: Task name

    Returns:
        Loss value
    """
    model.train()

    optimizer.zero_grad()

    logits = model(batch["inputs"], task_name=task)
    loss = criterion(logits, batch["mask"])

    loss.backward()

    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    optimizer.step()

    return loss.item()