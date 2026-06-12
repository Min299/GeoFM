"""geofm.evaluation.multitask_metrics

Metrics for multi-task learning.

Provides utilities for tracking and aggregating metrics across tasks.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import numpy as np


@dataclass
class TaskMetric:
    """Single metric for a task."""

    task_name: str
    metric_name: str
    value: float
    step: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "task": self.task_name,
            "metric": self.metric_name,
            "value": self.value,
            "step": self.step,
        }


@dataclass
class TaskMetrics:
    """Metrics for a single task across steps."""

    task_name: str
    losses: List[float] = field(default_factory=list)
    metrics: Dict[str, List[float]] = field(default_factory=dict)

    def add_loss(self, loss: float, step: int = 0) -> None:
        """Add a loss value.

        Args:
            loss: Loss value
            step: Step number
        """
        self.losses.append(loss)

    def add_metric(self, name: str, value: float, step: int = 0) -> None:
        """Add a metric value.

        Args:
            name: Metric name
            value: Metric value
            step: Step number
        """
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    def get_avg_loss(self) -> float:
        """Get average loss.

        Returns:
            Average loss
        """
        return np.mean(self.losses) if self.losses else 0.0

    def get_avg_metric(self, name: str) -> float:
        """Get average metric value.

        Args:
            name: Metric name

        Returns:
            Average metric value
        """
        if name not in self.metrics:
            return 0.0
        return np.mean(self.metrics[name]) if self.metrics[name] else 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "task": self.task_name,
            "avg_loss": self.get_avg_loss(),
            "final_loss": self.losses[-1] if self.losses else 0.0,
            "best_loss": min(self.losses) if self.losses else 0.0,
            "metrics": {
                name: {
                    "avg": self.get_avg_metric(name),
                    "final": self.metrics[name][-1] if self.metrics[name] else 0.0,
                    "best": max(self.metrics[name]) if self.metrics[name] else 0.0,
                }
                for name in self.metrics
            },
        }


class MultiTaskMetrics:
    """Container for metrics from multiple tasks.

    Usage:
        metrics = MultiTaskMetrics()
        metrics.update("flood", loss=0.5)
        metrics.update("burn", loss=0.6)
        summary = metrics.summary()
    """

    def __init__(self):
        """Initialize multi-task metrics."""
        self.task_metrics: Dict[str, TaskMetrics] = {}
        self.history: List[Dict] = []

    def update(
        self,
        task: str,
        loss: Optional[float] = None,
        step: int = 0,
        **kwargs,
    ) -> None:
        """Update metrics for a task.

        Args:
            task: Task name
            loss: Optional loss value
            step: Step number
            **kwargs: Additional metrics
        """
        if task not in self.task_metrics:
            self.task_metrics[task] = TaskMetrics(task_name=task)

        if loss is not None:
            self.task_metrics[task].add_loss(loss, step)

        for name, value in kwargs.items():
            self.task_metrics[task].add_metric(name, value, step)

    def summary(self) -> Dict[str, Any]:
        """Get summary of all metrics.

        Returns:
            Dictionary with metrics for all tasks
        """
        summary = {}
        for task, metrics in self.task_metrics.items():
            summary[task] = metrics.to_dict()

        return summary

    def get_task_metric(self, task: str) -> Optional[TaskMetrics]:
        """Get metrics for a specific task.

        Args:
            task: Task name

        Returns:
            TaskMetrics or None
        """
        return self.task_metrics.get(task)

    def get_avg_loss(self, task: str) -> float:
        """Get average loss for a task.

        Args:
            task: Task name

        Returns:
            Average loss
        """
        if task not in self.task_metrics:
            return 0.0
        return self.task_metrics[task].get_avg_loss()

    def get_all_losses(self) -> Dict[str, List[float]]:
        """Get all loss values per task.

        Returns:
            Dictionary mapping task to loss list
        """
        return {
            task: metrics.losses
            for task, metrics in self.task_metrics.items()
        }

    def tasks(self) -> List[str]:
        """Get list of tasks with metrics.

        Returns:
            List of task names
        """
        return list(self.task_metrics.keys())

    def to_dict(self) -> Dict:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "tasks": self.tasks(),
            "summary": self.summary(),
            "history": self.history,
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.task_metrics.clear()
        self.history.clear()


class AggregatedMetrics:
    """Aggregated metrics across multiple training runs."""

    def __init__(self):
        """Initialize aggregated metrics."""
        self.runs: List[Dict] = []

    def add_run(self, metrics: Dict) -> None:
        """Add metrics from a run.

        Args:
            metrics: Metrics dictionary
        """
        self.runs.append(metrics)

    def get_task_metric_across_runs(
        self,
        task: str,
        metric: str = "loss",
    ) -> Dict[str, float]:
        """Get metric across runs for a task.

        Args:
            task: Task name
            metric: Metric name

        Returns:
            Dictionary with mean, std, min, max
        """
        values = []
        for run in self.runs:
            if task in run and metric in run[task]:
                values.append(run[task][metric])

        if not values:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}

        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
        }

    def to_dict(self) -> Dict:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "num_runs": len(self.runs),
            "runs": self.runs,
        }


class MetricTracker:
    """Track metrics with moving averages and history."""

    def __init__(self, window_size: int = 100):
        """Initialize metric tracker.

        Args:
            window_size: Size of moving average window
        """
        self.window_size = window_size
        self.values: List[float] = []

    def add(self, value: float) -> None:
        """Add a value.

        Args:
            value: Metric value
        """
        self.values.append(value)

    def get_moving_avg(self) -> float:
        """Get moving average.

        Returns:
            Moving average of last window_size values
        """
        if not self.values:
            return 0.0

        window = self.values[-self.window_size:]
        return np.mean(window)

    def get_last(self) -> float:
        """Get last value.

        Returns:
            Most recent value
        """
        return self.values[-1] if self.values else 0.0

    def get_best(self) -> float:
        """Get best (minimum) value.

        Returns:
            Minimum value
        """
        return np.min(self.values) if self.values else 0.0

    def get_trend(self) -> str:
        """Get trend direction.

        Returns:
            "increasing", "decreasing", or "stable"
        """
        if len(self.values) < 10:
            return "stable"

        first_half = np.mean(self.values[:len(self.values)//2])
        second_half = np.mean(self.values[len(self.values)//2:])

        diff = second_half - first_half
        if diff > 0.01:
            return "increasing"
        elif diff < -0.01:
            return "decreasing"
        else:
            return "stable"