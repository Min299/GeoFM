"""geofm.losses.regression

Regression loss functions for GeoFM.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    """Mean squared error loss."""
    def forward(self, pred, target):
        return F.mse_loss(pred, target)


class MAELoss(nn.Module):
    """Mean absolute error loss."""
    def forward(self, pred, target):
        return F.l1_loss(pred, target)


class HuberLoss(nn.Module):
    """Huber loss - robust to outliers."""
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
    def forward(self, pred, target):
        return F.huber_loss(pred, target, delta=self.delta)


class SmoothL1Loss(nn.Module):
    """Smooth L1 loss."""
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
    def forward(self, pred, target):
        return F.smooth_l1_loss(pred, target, beta=self.beta)


class QuantileLoss(nn.Module):
    """Quantile regression loss."""
    def __init__(self, quantiles=(0.25, 0.5, 0.75)):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, pred, target):
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - pred[:, i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
        return torch.cat(losses, dim=1).mean()
