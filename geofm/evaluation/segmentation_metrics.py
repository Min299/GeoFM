"""geofm.evaluation.segmentation_metrics

Segmentation metrics for flood/burn segmentation evaluation.
"""
from __future__ import annotations

import torch


class SegmentationMetrics:
    """Computes segmentation metrics (IoU, Dice, F1, etc.).

    Usage:
        metrics = SegmentationMetrics()

        for batch in dataloader:
            logits, target = batch
            metrics.update(logits, target)

        results = metrics.compute()
        # {'iou': 0.85, 'dice': 0.92, 'precision': 0.90, ...}
    """

    def __init__(self, eps: float = 1e-6):
        """Initialize metrics tracker.

        Args:
            eps: Small value to prevent division by zero
        """
        self.eps = eps
        self.reset()

    def reset(self):
        """Reset accumulated statistics."""
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0

    @torch.no_grad()
    def update(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ):
        """Update metrics with a batch.

        Args:
            logits: Model output logits (B, C, H, W) or (B, H, W)
            target: Ground truth labels (B, H, W) or (H, W)
        """
        # Handle different input shapes
        if logits.dim() == 4:
            # (B, C, H, W) -> argmax to get class predictions
            pred = torch.argmax(logits, dim=1)
        else:
            pred = logits

        # Ensure target is same type
        if target.dim() > pred.dim():
            target = target.squeeze(1)

        pred = pred.bool()
        target = target.bool()

        # Update confusion matrix
        self.tp += (pred & target).sum().item()
        self.fp += (pred & ~target).sum().item()
        self.fn += (~pred & target).sum().item()
        self.tn += (~pred & ~target).sum().item()

    def compute(self) -> dict:
        """Compute all metrics.

        Returns:
            Dictionary with IoU, Dice, F1, precision, recall
        """
        # Precision
        precision = self.tp / (
            self.tp + self.fp + self.eps
        )

        # Recall
        recall = self.tp / (
            self.tp + self.fn + self.eps
        )

        # IoU (Jaccard)
        iou = self.tp / (
            self.tp + self.fp + self.fn + self.eps
        )

        # Dice (F1 for binary segmentation)
        dice = (
            2 * self.tp
        ) / (
            2 * self.tp + self.fp + self.fn + self.eps
        )

        # F1 score
        f1 = (
            2 * precision * recall
        ) / (
            precision + recall + self.eps
        )

        # Accuracy
        total = self.tp + self.fp + self.fn + self.tn
        accuracy = (self.tp + self.tn) / (total + self.eps)

        return {
            "iou": iou,
            "miou": iou,  # Alias for mean IoU (same for binary)
            "dice": dice,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
        }

    def __repr__(self):
        metrics = self.compute()
        return (
            f"SegmentationMetrics("
            f"IoU={metrics['iou']:.4f}, "
            f"Dice={metrics['dice']:.4f}, "
            f"F1={metrics['f1']:.4f})"
        )