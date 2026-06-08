"""geofm.experiments.adaptation_benchmark

Benchmark runner for adaptation methods.
"""
from __future__ import annotations

from typing import Optional, List

import torch


class AdaptationBenchmark:
    """Benchmark for comparing adaptation methods.

    Runs a single adaptation method on a task and collects metrics.

    Usage:
        benchmark = AdaptationBenchmark(config, model, trainer)
        history = benchmark.run(train_loader)
    """

    def __init__(
        self,
        config,
        model: torch.nn.Module,
        trainer,
        metrics=None,
    ):
        """Initialize benchmark.

        Args:
            config: BenchmarkConfig
            model: Model to train
            trainer: Trainer instance
            metrics: Optional metrics tracker
        """
        self.config = config
        self.model = model
        self.trainer = trainer
        self.metrics = metrics
        self.history = []

    def run(
        self,
        train_loader,
        val_loader=None,
    ) -> dict:
        """Run benchmark.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader

        Returns:
            Dictionary with training history and final metrics
        """
        self.history = []

        for epoch in range(self.config.epochs):
            # Train
            train_loss = self.trainer.train_epoch(
                train_loader,
                task=self.config.task,
            )

            self.history.append({
                "epoch": epoch,
                "train_loss": train_loss,
            })

            # Validate
            if val_loader is not None:
                val_loss = self.trainer.eval_epoch(
                    val_loader,
                    task=self.config.task,
                )
                self.history[-1]["val_loss"] = val_loss

        return {
            "experiment": self.config.experiment_name,
            "task": self.config.task,
            "epochs": self.config.epochs,
            "history": self.history,
        }

    def run_single_epoch(self, train_loader) -> float:
        """Run single training epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Mean training loss
        """
        return self.trainer.train_epoch(train_loader, task=self.config.task)

    def get_final_loss(self) -> float:
        """Get final training loss.

        Returns:
            Final loss from history
        """
        if not self.history:
            return float("inf")
        return self.history[-1].get("train_loss", float("inf"))

    def get_best_loss(self) -> float:
        """Get best (minimum) training loss.

        Returns:
            Minimum loss from history
        """
        if not self.history:
            return float("inf")
        losses = [h.get("train_loss", float("inf")) for h in self.history]
        return min(losses)


class MultiTaskBenchmark:
    """Benchmark for multitask learning.

    Runs multiple tasks and collects metrics.
    """

    def __init__(
        self,
        config,
        model: torch.nn.Module,
        trainer,
    ):
        """Initialize multi-task benchmark.

        Args:
            config: BenchmarkSuiteConfig
            model: Model to train
            trainer: Trainer instance
        """
        self.config = config
        self.model = model
        self.trainer = trainer
        self.results = {}

    def run_single_task(
        self,
        task: str,
        train_loader,
        val_loader=None,
    ) -> dict:
        """Run benchmark for single task.

        Args:
            task: Task name
            train_loader: Training data loader
            val_loader: Optional validation data loader

        Returns:
            Results dictionary
        """
        from geofm.experiments.benchmark_config import BenchmarkConfig

        task_config = BenchmarkConfig(
            experiment_name=self.config.experiment_name,
            task=task,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            output_dir=self.config.output_dir,
        )

        benchmark = AdaptationBenchmark(task_config, self.model, self.trainer)
        return benchmark.run(train_loader, val_loader)

    def run_all_tasks(
        self,
        task_loaders: dict,
        val_loaders: Optional[dict] = None,
    ) -> dict:
        """Run benchmark for all tasks.

        Args:
            task_loaders: Dict mapping task name to train loader
            val_loaders: Optional dict of validation loaders

        Returns:
            Results for all tasks
        """
        val_loaders = val_loaders or {}

        results = {}
        for task, train_loader in task_loaders.items():
            results[task] = self.run_single_task(
                task,
                train_loader,
                val_loaders.get(task),
            )

        self.results = results
        return results