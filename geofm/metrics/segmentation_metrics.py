"""geofm.metrics.segmentation_metrics

Segmentation metrics for evaluation.
"""
from __future__ import annotations

import torch


def pixel_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate pixel accuracy.

    Args:
        preds: Model predictions (B, C, H, W) or (B, H, W)
        targets: Ground truth labels (B, H, W)

    Returns:
        Pixel accuracy (0-1)
    """
    if preds.dim() == 4:
        preds = preds.argmax(dim=1)

    correct = (preds == targets).float()
    return correct.mean().item()


def dice_coefficient(preds: torch.Tensor, targets: torch.Tensor, num_classes: int = 2) -> float:
    """Calculate Dice coefficient.

    Args:
        preds: Model predictions (B, C, H, W) or (B, H, W)
        targets: Ground truth labels (B, H, W)
        num_classes: Number of classes

    Returns:
        Mean Dice coefficient (0-1)
    """
    if preds.dim() == 4:
        preds = preds.argmax(dim=1)

    dice_scores = []
    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        target_cls = (targets == cls).float()

        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()

        if union == 0:
            dice_scores.append(1.0)  # Empty class
        else:
            dice_scores.append((2.0 * intersection / union).item())

    return sum(dice_scores) / len(dice_scores)


def iou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int = 2) -> float:
    """Calculate Intersection over Union (IoU).

    Args:
        preds: Model predictions (B, C, H, W) or (B, H, W)
        targets: Ground truth labels (B, H, W)
        num_classes: Number of classes

    Returns:
        Mean IoU (0-1)
    """
    if preds.dim() == 4:
        preds = preds.argmax(dim=1)

    iou_scores = []
    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        target_cls = (targets == cls).float()

        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum() - intersection

        if union == 0:
            iou_scores.append(1.0)  # Empty class
        else:
            iou_scores.append((intersection / union).item())

    return sum(iou_scores) / len(iou_scores)


def precision(preds: torch.Tensor, targets: torch.Tensor, num_classes: int = 2) -> float:
    """Calculate precision.

    Args:
        preds: Model predictions
        targets: Ground truth labels
        num_classes: Number of classes

    Returns:
        Mean precision (0-1)
    """
    if preds.dim() == 4:
        preds = preds.argmax(dim=1)

    precision_scores = []
    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        target_cls = (targets == cls).float()

        tp = (pred_cls * target_cls).sum()
        fp = (pred_cls * (1 - target_cls)).sum()

        if tp + fp == 0:
            precision_scores.append(1.0)
        else:
            precision_scores.append((tp / (tp + fp)).item())

    return sum(precision_scores) / len(precision_scores)


def recall(preds: torch.Tensor, targets: torch.Tensor, num_classes: int = 2) -> float:
    """Calculate recall.

    Args:
        preds: Model predictions
        targets: Ground truth labels
        num_classes: Number of classes

    Returns:
        Mean recall (0-1)
    """
    if preds.dim() == 4:
        preds = preds.argmax(dim=1)

    recall_scores = []
    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        target_cls = (targets == cls).float()

        tp = (pred_cls * target_cls).sum()
        fn = ((1 - pred_cls) * target_cls).sum()

        if tp + fn == 0:
            recall_scores.append(1.0)
        else:
            recall_scores.append((tp / (tp + fn)).item())

    return sum(recall_scores) / len(recall_scores)


def f1_score(preds: torch.Tensor, targets: torch.Tensor, num_classes: int = 2) -> float:
    """Calculate F1 score.

    Args:
        preds: Model predictions
        targets: Ground truth labels
        num_classes: Number of classes

    Returns:
        Mean F1 score (0-1)
    """
    prec = precision(preds, targets, num_classes)
    rec = recall(preds, targets, num_classes)

    if prec + rec == 0:
        return 0.0

    return 2 * (prec * rec) / (prec + rec)


class SegmentationMetrics:
    """Container for segmentation metrics."""

    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
        self.reset()

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """Update metrics with new batch.

        Args:
            preds: Model predictions (B, C, H, W)
            targets: Ground truth (B, H, W)
        """
        self.accuracy += pixel_accuracy(preds, targets)
        self.dice += dice_coefficient(preds, targets, self.num_classes)
        self.iou += iou(preds, targets, self.num_classes)
        self.f1 += f1_score(preds, targets, self.num_classes)
        self.count += 1

    def compute(self) -> dict:
        """Compute mean metrics.

        Returns:
            Dictionary of metrics
        """
        if self.count == 0:
            return {}

        return {
            "accuracy": self.accuracy / self.count,
            "dice": self.dice / self.count,
            "iou": self.iou / self.count,
            "f1": self.f1 / self.count,
        }

    def reset(self):
        """Reset metrics."""
        self.accuracy = 0.0
        self.dice = 0.0
        self.iou = 0.0
        self.f1 = 0.0
        self.count = 0