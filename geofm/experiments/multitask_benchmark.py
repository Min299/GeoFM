"""geofm.experiments.multitask_benchmark

Multi-task benchmark runner.

Provides benchmarking utilities for multi-task learning experiments.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import time

from geofm.experiments.multitask_experiment import (
    MultiTaskExperiment,
    ExperimentConfig,
    ExperimentResult,
)
from geofm.tasks.task_scheduler import TaskScheduler
from geofm.training.multitask_trainer import MultiTaskTrainer
from geofm.evaluation.multitask_report import ComparisonReport


@dataclass
class BenchmarkTask:
    """Configuration for a benchmark task."""

    name: str
    weight: float = 1.0
    enabled: bool = True

    def to_dict(self) -> Dict:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "name": self.name,
            "weight": self.weight,
            "enabled": self.enabled,
        }


@dataclass
class MultiTaskBenchmarkConfig:
    """Configuration for multi-task benchmark."""

    adapter_type: str = "feature"
    tasks: List[BenchmarkTask] = field(default_factory=list)
    epochs: int = 10
    steps_per_epoch: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01

    @classmethod
    def from_tasks(
        cls,
        task_names: List[str],
        adapter_type: str = "feature",
        **kwargs,
    ) -> "MultiTaskBenchmarkConfig":
        """Create config from task names.

        Args:
            task_names: List of task names
            adapter_type: Adapter type
            **kwargs: Additional config parameters

        Returns:
            MultiTaskBenchmarkConfig
        """
        tasks = [BenchmarkTask(name=name) for name in task_names]
        return cls(adapter_type=adapter_type, tasks=tasks, **kwargs)

    def get_task_names(self) -> List[str]:
        """Get list of enabled task names.

        Returns:
            List of task names
        """
        return [t.name for t in self.tasks if t.enabled]

    def to_dict(self) -> Dict:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "adapter_type": self.adapter_type,
            "tasks": [t.to_dict() for t in self.tasks],
            "epochs": self.epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
        }


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""

    config: MultiTaskBenchmarkConfig
    results: Dict[str, Dict] = field(default_factory=dict)
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "config": self.config.to_dict(),
            "results": self.results,
            "duration_seconds": self.duration_seconds,
        }


class MultiTaskBenchmark:
    """Multi-task benchmark runner.

    Provides benchmarking for multi-task learning with multiple methods.

    Usage:
        benchmark = MultiTaskBenchmark()
        benchmark.add_config("feature", tasks=["flood", "burn"])
        benchmark.add_config("lora", tasks=["flood", "burn"])
        results = benchmark.run(model, optimizer, criterion, task_batcher)
    """

    def __init__(self):
        """Initialize multi-task benchmark."""
        self.configs: Dict[str, MultiTaskBenchmarkConfig] = {}
        self.results: Dict[str, BenchmarkResult] = {}

    def add_config(
        self,
        name: str,
        tasks: List[str],
        **kwargs,
    ) -> None:
        """Add a benchmark configuration.

        Args:
            name: Configuration name (e.g., "feature", "lora")
            tasks: List of task names
            **kwargs: Additional config parameters
        """
        self.configs[name] = MultiTaskBenchmarkConfig.from_tasks(
            task_names=tasks,
            **kwargs,
        )

    def add_benchmark_config(
        self,
        name: str,
        config: MultiTaskBenchmarkConfig,
    ) -> None:
        """Add a full benchmark config.

        Args:
            name: Configuration name
            config: MultiTaskBenchmarkConfig
        """
        self.configs[name] = config

    def run(
        self,
        model: Any,
        optimizer_factory: Any,
        criterion: Any,
        task_batcher: Any,
        device: Optional[Any] = None,
    ) -> Dict[str, BenchmarkResult]:
        """Run all benchmark configurations.

        Args:
            model: Model to train
            optimizer_factory: Factory function for creating optimizers
            criterion: Loss function
            task_batcher: TaskBatcher for data
            device: Optional device

        Returns:
            Dictionary mapping config name to results
        """
        for name, config in self.configs.items():
            print(f"Running benchmark: {name}")

            result = self._run_single(
                model=model,
                config=config,
                optimizer_factory=optimizer_factory,
                criterion=criterion,
                task_batcher=task_batcher,
                device=device,
            )

            self.results[name] = result

        return self.results

    def _run_single(
        self,
        model: Any,
        config: MultiTaskBenchmarkConfig,
        optimizer_factory: Any,
        criterion: Any,
        task_batcher: Any,
        device: Optional[Any] = None,
    ) -> BenchmarkResult:
        """Run a single benchmark configuration.

        Args:
            model: Model to train
            config: Benchmark config
            optimizer_factory: Factory function for optimizers
            criterion: Loss function
            task_batcher: TaskBatcher
            device: Optional device

        Returns:
            BenchmarkResult
        """
        start_time = time.time()

        # Create trainer
        tasks = config.get_task_names()
        scheduler = TaskScheduler(tasks)

        optimizer = optimizer_factory(model.parameters())

        trainer = MultiTaskTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            tasks=tasks,
            scheduler=scheduler,
            device=device,
        )

        # Run training
        results = {}

        for epoch in range(config.epochs):
            epoch_result = trainer.train_epoch(
                task_batcher,
                num_steps=config.steps_per_epoch,
            )

            results[epoch] = {
                "loss": epoch_result.avg_loss,
                "task_losses": epoch_result.task_losses,
            }

        duration = time.time() - start_time

        return BenchmarkResult(
            config=config,
            results=results,
            duration_seconds=duration,
        )

    def get_results(self) -> Dict[str, BenchmarkResult]:
        """Get all benchmark results.

        Returns:
            Dictionary of results
        """
        return self.results

    def get_comparison_report(self) -> ComparisonReport:
        """Get comparison report for all results.

        Returns:
            ComparisonReport
        """
        report = ComparisonReport()

        for name, result in self.results.items():
            # Convert results to metrics format
            metrics = {}

            for epoch, epoch_data in result.results.items():
                for task, losses in epoch_data.get("task_losses", {}).items():
                    if task not in metrics:
                        metrics[task] = {"losses": [], "avg_loss": 0}

                    metrics[task]["losses"].extend(losses)

            # Compute average losses
            for task, task_metrics in metrics.items():
                losses = task_metrics["losses"]
                task_metrics["avg_loss"] = sum(losses) / len(losses) if losses else 0

            report.add_experiment(name, metrics)

        return report

    def get_best_method(
        self,
        task: str,
        metric: str = "avg_loss",
    ) -> Optional[str]:
        """Get best method for a task.

        Args:
            task: Task name
            metric: Metric to optimize

        Returns:
            Best method name or None
        """
        comparison = self.get_comparison_report()
        return comparison.get_best_experiment(task, metric)

    def to_dict(self) -> Dict:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "configs": {
                name: config.to_dict()
                for name, config in self.configs.items()
            },
            "results": {
                name: result.to_dict()
                for name, result in self.results.items()
            },
        }


class QuickMultiTaskBenchmark(MultiTaskBenchmark):
    """Quick benchmark for smoke testing.

    Runs with reduced epochs for faster testing.
    """

    def __init__(self, quick_epochs: int = 2, quick_steps: int = 10):
        """Initialize quick benchmark.

        Args:
            quick_epochs: Number of epochs for quick run
            quick_steps: Number of steps per epoch
        """
        super().__init__()
        self.quick_epochs = quick_epochs
        self.quick_steps = quick_steps

    def run(
        self,
        model: Any,
        optimizer_factory: Any,
        criterion: Any,
        task_batcher: Any,
        device: Optional[Any] = None,
    ) -> Dict[str, BenchmarkResult]:
        """Run quick benchmark.

        Args:
            model: Model to train
            optimizer_factory: Factory function
            criterion: Loss function
            task_batcher: TaskBatcher
            device: Optional device

        Returns:
            Dictionary of results
        """
        # Override configs with quick settings
        for config in self.configs.values():
            config.epochs = self.quick_epochs
            config.steps_per_epoch = self.quick_steps

        return super().run(
            model=model,
            optimizer_factory=optimizer_factory,
            criterion=criterion,
            task_batcher=task_batcher,
            device=device,
        )