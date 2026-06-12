"""geofm.logging.metric_logger

Metric tracking and logging.

Provides utilities for tracking training metrics.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Any
from collections import defaultdict
import numpy as np


class MetricLogger:
    """Logger for tracking metrics over time.

    Stores metric history and computes statistics.

    Usage:
        metrics = MetricLogger()
        metrics.update("loss", 0.5)
        metrics.update("loss", 0.4)
        metrics.update("accuracy", 0.8)

        latest = metrics.latest("loss")  # 0.4
        summary = metrics.summary()  # {"loss": 0.45, "accuracy": 0.8}
    """

    def __init__(self):
        """Initialize metric logger."""
        self.history: Dict[str, List[float]] = defaultdict(list)
        self.step: int = 0

    def update(
        self,
        key: str,
        value: float,
        step: Optional[int] = None,
    ) -> None:
        """Update a metric.

        Args:
            key: Metric name
            value: Metric value
            step: Optional step number
        """
        self.history[key].append(float(value))

        if step is not None:
            self.step = step
        else:
            self.step += 1

    def latest(self, key: str) -> Optional[float]:
        """Get latest value for a metric.

        Args:
            key: Metric name

        Returns:
            Latest value or None if not found
        """
        if key not in self.history or not self.history[key]:
            return None
        return self.history[key][-1]

    def mean(self, key: str) -> float:
        """Get mean of a metric.

        Args:
            key: Metric name

        Returns:
            Mean value or 0 if not found
        """
        if key not in self.history or not self.history[key]:
            return 0.0
        return float(np.mean(self.history[key]))

    def std(self, key: str) -> float:
        """Get standard deviation of a metric.

        Args:
            key: Metric name

        Returns:
            Standard deviation or 0 if not found
        """
        if key not in self.history or len(self.history[key]) < 2:
            return 0.0
        return float(np.std(self.history[key]))

    def min(self, key: str) -> float:
        """Get minimum of a metric.

        Args:
            key: Metric name

        Returns:
            Minimum value or 0 if not found
        """
        if key not in self.history or not self.history[key]:
            return 0.0
        return float(np.min(self.history[key]))

    def max(self, key: str) -> float:
        """Get maximum of a metric.

        Args:
            key: Metric name

        Returns:
            Maximum value or 0 if not found
        """
        if key not in self.history or not self.history[key]:
            return 0.0
        return float(np.max(self.history[key]))

    def summary(self) -> Dict[str, float]:
        """Get summary statistics for all metrics.

        Returns:
            Dictionary mapping metric to mean value
        """
        return {
            key: self.mean(key)
            for key in self.history
        }

    def detailed_summary(self) -> Dict[str, Dict[str, float]]:
        """Get detailed summary for all metrics.

        Returns:
            Dictionary mapping metric to stats dict
        """
        summary = {}

        for key in self.history:
            summary[key] = {
                "mean": self.mean(key),
                "std": self.std(key),
                "min": self.min(key),
                "max": self.max(key),
                "latest": self.latest(key),
                "count": len(self.history[key]),
            }

        return summary

    def get_history(self, key: str) -> List[float]:
        """Get full history for a metric.

        Args:
            key: Metric name

        Returns:
            List of values
        """
        return self.history.get(key, [])

    def get_recent(
        self,
        key: str,
        n: int = 10,
    ) -> List[float]:
        """Get recent values for a metric.

        Args:
            key: Metric name
            n: Number of recent values

        Returns:
            List of recent values
        """
        history = self.history.get(key, [])
        return history[-n:]

    def get_moving_avg(
        self,
        key: str,
        window: int = 10,
    ) -> float:
        """Get moving average of a metric.

        Args:
            key: Metric name
            window: Window size

        Returns:
            Moving average
        """
        history = self.history.get(key, [])
        if not history:
            return 0.0

        recent = history[-window:]
        return float(np.mean(recent))

    def has_metric(self, key: str) -> bool:
        """Check if a metric exists.

        Args:
            key: Metric name

        Returns:
            True if metric exists
        """
        return key in self.history and len(self.history[key]) > 0

    def clear(self) -> None:
        """Clear all metrics."""
        self.history.clear()
        self.step = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "history": dict(self.history),
            "step": self.step,
        }


class TaskMetricLogger(MetricLogger):
    """Metric logger that separates metrics by task.

    Useful for multi-task learning.
    """

    def update(
        self,
        key: str,
        value: float,
        task: Optional[str] = None,
        step: Optional[int] = None,
    ) -> None:
        """Update a metric.

        Args:
            key: Metric name
            value: Metric value
            task: Optional task name
            step: Optional step number
        """
        full_key = f"{task}_{key}" if task else key
        super().update(full_key, value, step)

    def get_task_metrics(self, task: str) -> Dict[str, float]:
        """Get all metrics for a task.

        Args:
            task: Task name

        Returns:
            Dictionary of metrics for the task
        """
        prefix = f"{task}_"
        return {
            key.replace(prefix, ""): self.latest(key)
            for key in self.history
            if key.startswith(prefix)
        }

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary organized by task.

        Returns:
            Dictionary mapping task to metrics dict
        """
        tasks = set()
        for key in self.history:
            if "_" in key:
                task = key.rsplit("_", 1)[0]
                tasks.add(task)

        summary = {}
        for task in tasks:
            summary[task] = self.get_task_metrics(task)

        return summary