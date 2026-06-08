"""geofm.evaluation.classification_metrics

Classification metrics for LULC and other classification tasks.
"""
from __future__ import annotations

import torch


class ClassificationMetrics:
    """Computes classification metrics (accuracy, per-class).

    Usage:
        metrics = ClassificationMetrics()

        for batch in dataloader:
            logits, target = batch
            metrics.update(logits, target)

        results = metrics.compute()
        # {'accuracy': 0.85, 'per_class_accuracy': [...]}
    """

    def __init__(self):
        """Initialize metrics tracker."""
        self.reset()

    def reset(self):
        """Reset accumulated statistics."""
        self.correct = 0
        self.total = 0
        self.class_correct = {}
        self.class_total = {}

    @torch.no_grad()
    def update(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ):
        """Update metrics with a batch.

        Args:
            logits: Model output logits (B, C)
            target: Ground truth labels (B,)
        """
        pred = logits.argmax(dim=1)

        # Overall accuracy
        self.correct += (pred == target).sum().item()
        self.total += target.numel()

        # Per-class accuracy
        for cls in target.unique().tolist():
            cls_mask = target == cls
            cls_correct = (pred[cls_mask] == target[cls_mask]).sum().item()
            cls_total = cls_mask.sum().item()

            if cls not in self.class_total:
                self.class_correct[cls] = 0
                self.class_total[cls] = 0

            self.class_correct[cls] += cls_correct
            self.class_total[cls] += cls_total

    def compute(self) -> dict:
        """Compute all metrics.

        Returns:
            Dictionary with accuracy, per-class accuracy
        """
        acc = self.correct / max(self.total, 1)

        result = {
            "accuracy": acc,
            "correct": self.correct,
            "total": self.total,
        }

        # Per-class accuracy
        if self.class_total:
            per_class_acc = {}
            for cls in sorted(self.class_total.keys()):
                if self.class_total[cls] > 0:
                    per_class_acc[int(cls)] = (
                        self.class_correct[cls] / self.class_total[cls]
                    )
                else:
                    per_class_acc[int(cls)] = 0.0

            result["per_class_accuracy"] = per_class_acc
            result["mean_class_accuracy"] = (
                sum(per_class_acc.values()) / len(per_class_acc)
                if per_class_acc else 0.0
            )

        return result

    def __repr__(self):
        metrics = self.compute()
        return (
            f"ClassificationMetrics("
            f"accuracy={metrics['accuracy']:.4f})"
        )