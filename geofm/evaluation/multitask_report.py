"""geofm.evaluation.multitask_report

Reporting utilities for multi-task learning.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class TaskResult:
    """Result for a single task."""

    task_name: str
    final_loss: float
    best_loss: float
    final_metric: float = 0.0
    best_metric: float = 0.0
    epochs_trained: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "task": self.task_name,
            "final_loss": self.final_loss,
            "best_loss": self.best_loss,
            "final_metric": self.final_metric,
            "best_metric": self.best_metric,
            "epochs_trained": self.epochs_trained,
        }


class MultiTaskReport:
    """Report for multi-task learning results.

    Usage:
        report = MultiTaskReport(metrics)
        report.print_report()
        table = report.get_table()
    """

    def __init__(
        self,
        metrics: Dict[str, Any],
        title: str = "Multi-Task Results",
    ):
        """Initialize multi-task report.

        Args:
            metrics: Dictionary mapping task to metrics
            title: Report title
        """
        self.metrics = metrics
        self.title = title

    def print_report(self) -> None:
        """Print report to console."""
        print(f"\n{'=' * 60}")
        print(f"{self.title}")
        print(f"{'=' * 60}\n")

        for task, task_metrics in self.metrics.items():
            self._print_task(task, task_metrics)

        print(f"\n{'=' * 60}\n")

    def _print_task(self, task: str, metrics: Dict) -> None:
        """Print single task metrics.

        Args:
            task: Task name
            metrics: Task metrics dictionary
        """
        print(f"Task: {task}")
        print("-" * 40)

        if "avg_loss" in metrics:
            print(f"  Avg Loss:    {metrics['avg_loss']:.4f}")
        if "final_loss" in metrics:
            print(f"  Final Loss:  {metrics['final_loss']:.4f}")
        if "best_loss" in metrics:
            print(f"  Best Loss:   {metrics['best_loss']:.4f}")

        if "metrics" in metrics:
            for name, values in metrics["metrics"].items():
                if isinstance(values, dict):
                    print(f"  {name}:")
                    for k, v in values.items():
                        print(f"    {k}: {v:.4f}")

        print()

    def get_table(self) -> str:
        """Get results as a formatted table.

        Returns:
            Formatted table string
        """
        lines = []
        lines.append(f"{'Task':<15} {'Final Loss':<12} {'Best Loss':<12} {'Avg Metric':<12}")
        lines.append("-" * 60)

        for task, metrics in self.metrics.items():
            final = metrics.get("final_loss", metrics.get("avg_loss", 0))
            best = metrics.get("best_loss", 0)
            avg_metric = metrics.get("final_metric", 0)

            lines.append(
                f"{task:<15} {final:<12.4f} {best:<12.4f} {avg_metric:<12.4f}"
            )

        return "\n".join(lines)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics.

        Returns:
            Dictionary with summary
        """
        total_loss = 0
        count = 0

        for task, metrics in self.metrics.items():
            if "avg_loss" in metrics:
                total_loss += metrics["avg_loss"]
                count += 1

        return {
            "num_tasks": len(self.metrics),
            "avg_loss": total_loss / count if count > 0 else 0,
            "tasks": list(self.metrics.keys()),
        }

    def to_dict(self) -> Dict:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "title": self.title,
            "metrics": self.metrics,
            "summary": self.get_summary(),
        }


class ComparisonReport:
    """Report for comparing multiple experiments."""

    def __init__(self):
        """Initialize comparison report."""
        self.experiments: Dict[str, Dict] = {}

    def add_experiment(
        self,
        name: str,
        metrics: Dict[str, Any],
    ) -> None:
        """Add an experiment.

        Args:
            name: Experiment name
            metrics: Metrics dictionary
        """
        self.experiments[name] = metrics

    def print_comparison(self) -> None:
        """Print comparison report."""
        print("\n" + "=" * 60)
        print("Experiment Comparison")
        print("=" * 60 + "\n")

        # Get all tasks
        tasks = set()
        for exp in self.experiments.values():
            tasks.update(exp.keys())

        # Print comparison for each task
        for task in sorted(tasks):
            print(f"Task: {task}")
            print("-" * 40)

            for name, metrics in self.experiments.items():
                if task in metrics:
                    loss = metrics[task].get("avg_loss", 0)
                    print(f"  {name}: {loss:.4f}")

            print()

    def get_task_ranking(
        self,
        task: str,
        metric: str = "avg_loss",
    ) -> List[tuple]:
        """Get ranking of experiments for a task.

        Args:
            task: Task name
            metric: Metric to rank by

        Returns:
            List of (experiment_name, value) tuples
        """
        ranked = []

        for name, metrics in self.experiments.items():
            if task in metrics:
                value = metrics[task].get(metric, 0)
                ranked.append((name, value))

        # Sort by value (lower is better for loss)
        return sorted(ranked, key=lambda x: x[1])

    def get_best_experiment(
        self,
        task: str,
        metric: str = "avg_loss",
    ) -> Optional[str]:
        """Get best experiment for a task.

        Args:
            task: Task name
            metric: Metric to optimize

        Returns:
            Best experiment name or None
        """
        ranking = self.get_task_ranking(task, metric)
        return ranking[0][0] if ranking else None

    def to_dict(self) -> Dict:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "num_experiments": len(self.experiments),
            "experiments": self.experiments,
        }


class TrainingProgressReport:
    """Report for tracking training progress."""

    def __init__(self):
        """Initialize training progress report."""
        self.epochs: List[Dict] = []

    def add_epoch(
        self,
        epoch: int,
        losses: Dict[str, float],
        metrics: Optional[Dict[str, Dict]] = None,
    ) -> None:
        """Add epoch data.

        Args:
            epoch: Epoch number
            losses: Dictionary mapping task to loss
            metrics: Optional additional metrics
        """
        self.epochs.append({
            "epoch": epoch,
            "losses": losses,
            "metrics": metrics or {},
        })

    def print_progress(self) -> None:
        """Print training progress."""
        print("\nTraining Progress")
        print("-" * 60)
        print(f"{'Epoch':<8} {'Avg Loss':<12} {'Flood':<10} {'Burn':<10} {'LULC':<10}")
        print("-" * 60)

        for entry in self.epochs:
            epoch = entry["epoch"]
            losses = entry["losses"]

            avg_loss = sum(losses.values()) / len(losses) if losses else 0
            flood = losses.get("flood", 0)
            burn = losses.get("burn", 0)
            lulc = losses.get("lulc", 0)

            print(f"{epoch:<8} {avg_loss:<12.4f} {flood:<10.4f} {burn:<10.4f} {lulc:<10.4f}")

    def get_epoch_count(self) -> int:
        """Get number of epochs.

        Returns:
            Epoch count
        """
        return len(self.epochs)

    def get_last_losses(self) -> Dict[str, float]:
        """Get losses from last epoch.

        Returns:
            Dictionary mapping task to loss
        """
        if not self.epochs:
            return {}
        return self.epochs[-1]["losses"]

    def to_dict(self) -> Dict:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "num_epochs": len(self.epochs),
            "epochs": self.epochs,
        }