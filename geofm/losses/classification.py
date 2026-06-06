"""geofm.losses.classification

Classification loss functions for GeoFM.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing."""

    def __init__(self, smoothing=0.1, ignore_index=-1):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        n_classes = pred.shape[-1]
        log_preds = F.log_softmax(pred, dim=-1)
        mask = target != self.ignore_index
        target_filtered = target[mask]
        log_preds_filtered = log_preds[mask]
        if len(target_filtered) == 0:
            return torch.tensor(0.0, device=pred.device)
        nll = F.nll_loss(log_preds_filtered, target_filtered, reduction="mean")
        smooth_loss = -log_preds_filtered.mean()
        return (1 - self.smoothing) * nll + self.smoothing * smooth_loss


class WeightedCrossEntropy(nn.Module):
    """Weighted cross entropy for imbalanced classification."""

    def __init__(self, weights=None, ignore_index=-1):
        super().__init__()
        self.weights = weights
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        if self.weights is not None:
            self.weights = self.weights.to(pred.device)
        return F.cross_entropy(pred, target, weight=self.weights, ignore_index=self.ignore_index)


class BCEWithLogitsLoss(nn.Module):
    """Binary cross entropy for multi-label classification."""

    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, pred, target):
        if self.pos_weight is not None:
            self.pos_weight = self.pos_weight.to(pred.device)
        return F.binary_cross_entropy_with_logits(pred, target, pos_weight=self.pos_weight)
