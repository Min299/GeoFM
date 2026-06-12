"""geofm.logging.metrics_logger

Metrics logger for tracking training progress.
"""
from __future__ import annotations

from typing import Optional, Any


class MetricsLogger:
    """Logger for tracking metrics during training.

    Usage:
        logger = MetricsLogger()

        logger.log(epoch=1, split="train", metrics={"loss": 0.5, "iou": 0.7})
        logger.log(epoch=1, split="val", metrics={"loss": 0.6, "iou": 0.65})

        print(logger.latest())
        # {'epoch': 1, 'split': 'val', 'loss': 0.6, 'iou': 0.65}

        print(logger.latest("train"))
        # {'epoch': 1, 'split': 'train', 'loss': 0.5, 'iou': 0.7}
    """

    def __init__(self):
        """Initialize metrics logger."""
        self.history = []

    def log(
        self,
        epoch: int,
        split: str,
        metrics: dict,
    ):
        """Log metrics for an epoch.

        Args:
            epoch: Epoch number
            split: Data split ("train", "val", "test")
            metrics: Dictionary of metric values
        """
        row = {
            "epoch": epoch,
            "split": split,
        }

        row.update(metrics)

        self.history.append(row)

    def latest(
        self,
        split: Optional[str] = None,
    ) -> Optional[dict]:
        """Get the latest logged metrics.

        Args:
            split: If provided, get latest for this split only

        Returns:
            Latest metrics dict or None if empty
        """
        if not self.history:
            return None

        if split is None:
            return self.history[-1]

        # Find latest for this split
        for row in reversed(self.history):
            if row.get("split") == split:
                return row

        return None

    def get_history(
        self,
        split: Optional[str] = None,
    ) -> list:
        """Get full history or filtered by split.

        Args:
            split: If provided, filter to this split only

        Returns:
            List of metric dicts
        """
        if split is None:
            return self.history.copy()

        return [row for row in self.history if row.get("split") == split]

    def get_best(
        self,
        metric: str,
        split: Optional[str] = None,
        mode: str = "max",
    ) -> Optional[dict]:
        """Get best epoch for a metric.

        Args:
            metric: Metric name to optimize
            split: If provided, consider only this split
            mode: "max" for maximizing, "min" for minimizing

        Returns:
            Best row dict or None if empty
        """
        history = self.get_history(split)

        if not history:
            return None

        best = None
        best_value = float("-inf") if mode == "max" else float("inf")

        for row in history:
            value = row.get(metric)
            if value is None:
                continue

            if mode == "max" and value > best_value:
                best_value = value
                best = row
            elif mode == "min" and value < best_value:
                best_value = value
                best = row

        return best

    def to_dict(self) -> dict:
        """Convert history to dictionary for serialization.

        Returns:
            Dictionary with history list
        """
        return {"history": self.history}

    @classmethod
    def from_dict(cls, data: dict) -> "MetricsLogger":
        """Create logger from serialized dictionary.

        Args:
            data: Dictionary from to_dict()

        Returns:
            MetricsLogger with restored history
        """
        logger = cls()
        logger.history = data.get("history", [])
        return logger

    def __len__(self):
        return len(self.history)

    def __repr__(self):
        return f"MetricsLogger(history={len(self.history)} entries)"