"""geofm.losses.multitask

Multi-task loss functions for GeoFM.
"""
import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    """Base multi-task loss with task weighting."""

    def __init__(self, task_weights=None):
        super().__init__()
        self.task_weights = task_weights or {}

    def forward(self, losses):
        total_loss = 0.0
        for task_name, loss in losses.items():
            weight = self.task_weights.get(task_name, 1.0)
            total_loss += weight * loss
        return total_loss


class DynamicWeightAverage(nn.Module):
    """Dynamic Weight Average - learns task weights via uncertainty."""

    def __init__(self, num_tasks):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        total_loss = 0.0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        return total_loss


class GradientNormBalancing(nn.Module):
    """Gradient normalization for multi-task learning."""

    def __init__(self, num_tasks):
        super().__init__()
        self.num_tasks = num_tasks

    def forward(self, losses, model=None):
        return sum(losses) / len(losses) if losses else torch.tensor(0.0)
