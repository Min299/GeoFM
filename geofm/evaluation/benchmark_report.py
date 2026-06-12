"""geofm.evaluation.benchmark_report

Reporting utilities for benchmark results.
"""
from __future__ import annotations

from typing import Dict, List, Optional
import time


class BenchmarkReport:
    """Report for benchmark results.

    Generates formatted reports from benchmark results.
    """

    def __init__(
        self,
        results: Dict,
        title: str = "Benchmark Results",
    ):
        """Initialize report.

        Args:
            results: Dictionary of experiment results
            title: Report title
        """
        self.results = results
        self.title = title

    def print_report(self) -> None:
        """Print report to console."""
        print(f"\n{'=' * 60}")
        print(f"{self.title}")
        print(f"{'=' * 60}\n")

        for name, metrics in self.results.items():
            self._print_experiment(name, metrics)

        print(f"\n{'=' * 60}\n")

    def _print_experiment(self, name: str, metrics: dict) -> None:
        """Print single experiment results.

        Args:
            name: Experiment name
            metrics: Metrics dictionary
        """
        print(f"Experiment: {name}")
        print("-" * 40)

        if "dice" in metrics:
            print(f"  Dice:      {metrics['dice']:.4f}")
        if "iou" in metrics:
            print(f"  IoU:       {metrics['iou']:.4f}")
        if "f1" in metrics:
            print(f"  F1:        {metrics['f1']:.4f}")
        if "precision" in metrics:
            print(f"  Precision: {metrics['precision']:.4f}")
        if "recall" in metrics:
            print(f"  Recall:    {metrics['recall']:.4f}")
        if "train_time" in metrics:
            print(f"  Time:      {metrics['train_time']:.2f}s")
        if "trainable_params" in metrics:
            print(f"  Trainable: {metrics['trainable_params']:,}")
        if "total_params" in metrics:
            print(f"  Total:     {metrics['total_params']:,}")
        if "final_loss" in metrics:
            print(f"  Loss:      {metrics['final_loss']:.4f}")

        print()

    def get_summary_table(self) -> str:
        """Get results as a table string.

        Returns:
            Formatted table string
        """
        lines = []
        lines.append(f"{'Experiment':<15} {'Dice':<8} {'IoU':<8} {'F1':<8} {'Params':<12}")
        lines.append("-" * 60)

        for name, metrics in self.results.items():
            dice = metrics.get("dice", 0)
            iou = metrics.get("iou", 0)
            f1 = metrics.get("f1", 0)
            params = metrics.get("trainable_params", 0)

            lines.append(
                f"{name:<15} {dice:<8.4f} {iou:<8.4f} {f1:<8.4f} {params:<12,}"
            )

        return "\n".join(lines)

    def get_comparison_dict(self) -> dict:
        """Get comparison dictionary for ranking.

        Returns:
            Dictionary with experiment comparisons
        """
        comparison = {}

        for name, metrics in self.results.items():
            comparison[name] = {
                "dice": metrics.get("dice", 0),
                "iou": metrics.get("iou", 0),
                "f1": metrics.get("f1", 0),
                "trainable_ratio": metrics.get("trainable_ratio", 0),
                "train_time": metrics.get("train_time", 0),
            }

        return comparison


class MultiTaskReport:
    """Report for multi-task benchmark results."""

    def __init__(self, results: Dict):
        """Initialize multi-task report.

        Args:
            results: Dictionary mapping task to experiment results
        """
        self.results = results

    def print_report(self) -> None:
        """Print multi-task report."""
        print(f"\n{'=' * 60}")
        print("Multi-Task Benchmark Results")
        print(f"{'=' * 60}\n")

        for task, task_results in self.results.items():
            print(f"Task: {task}")
            print("-" * 40)

            report = BenchmarkReport(task_results)
            report.print_report()

        print(f"\n{'=' * 60}\n")


class ExperimentComparison:
    """Compare multiple experiments."""

    def __init__(self, results: Dict):
        """Initialize comparison.

        Args:
            results: Dictionary of experiment results
        """
        self.results = results

    def rank_by_metric(self, metric: str, descending: bool = True) -> List[tuple]:
        """Rank experiments by a metric.

        Args:
            metric: Metric to rank by
            descending: If True, higher is better

        Returns:
            List of (name, value) tuples sorted by metric
        """
        ranked = []
        for name, metrics in self.results.items():
            value = metrics.get(metric, 0)
            ranked.append((name, value))

        return sorted(ranked, key=lambda x: x[1], reverse=descending)

    def print_ranking(self, metric: str, descending: bool = True) -> None:
        """Print ranking for a metric.

        Args:
            metric: Metric to rank by
            descending: If True, higher is better
        """
        ranked = self.rank_by_metric(metric, descending)

        print(f"\nRanking by {metric}:")
        print("-" * 40)
        for i, (name, value) in enumerate(ranked, 1):
            print(f"  {i}. {name}: {value:.4f}")

    def get_best(self, metric: str, descending: bool = True) -> tuple:
        """Get best experiment for a metric.

        Args:
            metric: Metric to check
            descending: If True, higher is better

        Returns:
            Tuple of (name, value)
        """
        ranked = self.rank_by_metric(metric, descending)
        return ranked[0] if ranked else (None, None)