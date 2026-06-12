"""geofm.evaluation.benchmark_metrics

Metrics collected during benchmark runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BenchmarkMetrics:
    """Metrics collected during benchmark.

    Attributes:
        train_time: Total training time in seconds
        dice: Dice coefficient (0-1, higher is better)
        iou: Intersection over Union (0-1, higher is better)
        f1: F1 score (0-1, higher is better)
        precision: Precision (0-1)
        recall: Recall (0-1)
        trainable_params: Number of trainable parameters
        total_params: Total number of parameters
        final_loss: Final training loss
        best_loss: Best (minimum) training loss
        epochs_trained: Number of epochs completed
    """

    train_time: float = 0.0
    dice: float = 0.0
    iou: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    trainable_params: int = 0
    total_params: int = 0
    final_loss: float = float("inf")
    best_loss: float = float("inf")
    epochs_trained: int = 0

    # Extended metrics
    pixel_accuracy: float = 0.0
    trainable_ratio: float = 0.0
    memory_mb: float = 0.0
    throughput_samples_per_sec: float = 0.0

    # Per-epoch history
    history: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary.

        Returns:
            Metrics as dictionary
        """
        return {
            "train_time": self.train_time,
            "dice": self.dice,
            "iou": self.iou,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "pixel_accuracy": self.pixel_accuracy,
            "trainable_params": self.trainable_params,
            "total_params": self.total_params,
            "trainable_ratio": self.trainable_ratio,
            "final_loss": self.final_loss,
            "best_loss": self.best_loss,
            "epochs_trained": self.epochs_trained,
            "memory_mb": self.memory_mb,
            "throughput_samples_per_sec": self.throughput_samples_per_sec,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BenchmarkMetrics":
        """Create from dictionary.

        Args:
            data: Dictionary with metrics

        Returns:
            BenchmarkMetrics instance
        """
        return cls(**data)

    def update_from_history(self, history: list) -> None:
        """Update metrics from training history.

        Args:
            history: List of epoch metrics
        """
        self.history = history
        if history:
            self.epochs_trained = len(history)
            self.final_loss = history[-1].get("train_loss", float("inf"))
            self.best_loss = min(
                h.get("train_loss", float("inf")) for h in history
            )

    def compute_param_ratio(self) -> None:
        """Compute trainable parameter ratio."""
        if self.total_params > 0:
            self.trainable_ratio = self.trainable_params / self.total_params


@dataclass
class TaskMetrics:
    """Metrics for a single task."""

    task_name: str
    metrics: BenchmarkMetrics

    def to_dict(self) -> dict:
        """Convert to dictionary.

        Returns:
            Task metrics as dictionary
        """
        return {
            "task": self.task_name,
            "metrics": self.metrics.to_dict(),
        }


@dataclass
class ExperimentMetrics:
    """Metrics for a full experiment with multiple tasks."""

    experiment_name: str
    adapter_type: str
    task_metrics: dict[str, TaskMetrics] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary.

        Returns:
            Experiment metrics as dictionary
        """
        return {
            "experiment": self.experiment_name,
            "adapter_type": self.adapter_type,
            "tasks": {
                task: tm.to_dict()
                for task, tm in self.task_metrics.items()
            },
        }

    def add_task_metrics(self, task_name: str, metrics: BenchmarkMetrics) -> None:
        """Add metrics for a task.

        Args:
            task_name: Name of task
            metrics: Metrics for task
        """
        self.task_metrics[task_name] = TaskMetrics(task_name, metrics)