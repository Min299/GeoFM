"""geofm.training.predict

Prediction utilities for inference.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def predict(
    model,
    batch,
    task: str = "flood",
):
    """Run prediction on a batch.

    Args:
        model: The PyTorch model
        batch: Input batch (dict with 'inputs' key or tensor)
        task: Task name for model forward pass

    Returns:
        Model output logits
    """
    model.eval()

    # Handle both dict and tensor inputs
    if isinstance(batch, dict):
        inputs = batch.get("inputs", batch.get("image", batch.get("x")))
    else:
        inputs = batch

    return model(inputs, task_name=task)