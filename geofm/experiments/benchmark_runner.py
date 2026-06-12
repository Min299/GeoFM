"""geofm.experiments.benchmark_runner

Runner for executing benchmarks.
"""
from __future__ import annotations

from typing import Optional, List, Dict

import torch


class BenchmarkRunner:
    """Runner for executing benchmark experiments.

    Usage:
        runner = BenchmarkRunner(benchmark)
        results = runner.run(train_loader)
    """

    def __init__(
        self,
        benchmark,
    ):
        """Initialize runner.

        Args:
            benchmark: AdaptationBenchmark instance
        """
        self.benchmark = benchmark

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
            Results dictionary
        """
        return self.benchmark.run(train_loader, val_loader)

    def run_with_checkpointing(
        self,
        train_loader,
        val_loader=None,
        checkpoint_dir: Optional[str] = None,
        save_interval: int = 5,
    ) -> dict:
        """Run benchmark with checkpointing.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            checkpoint_dir: Directory to save checkpoints
            save_interval: Save every N epochs

        Returns:
            Results dictionary
        """
        results = {"checkpoints": []}

        for epoch in range(self.benchmark.config.epochs):
            # Train
            train_loss = self.benchmark.trainer.train_epoch(
                train_loader,
                task=self.benchmark.config.task,
            )

            self.benchmark.history.append({
                "epoch": epoch,
                "train_loss": train_loss,
            })

            # Validate
            if val_loader is not None:
                val_loss = self.benchmark.trainer.eval_epoch(
                    val_loader,
                    task=self.benchmark.config.task,
                )
                self.benchmark.history[-1]["val_loss"] = val_loss

            # Save checkpoint
            if checkpoint_dir and (epoch + 1) % save_interval == 0:
                checkpoint_path = f"{checkpoint_dir}/epoch_{epoch+1}.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.benchmark.model.state_dict(),
                    "train_loss": train_loss,
                }, checkpoint_path)
                results["checkpoints"].append(checkpoint_path)

        results["experiment"] = self.benchmark.config.experiment_name
        results["task"] = self.benchmark.config.task
        results["history"] = self.benchmark.history

        return results


class SuiteRunner:
    """Runner for executing multiple benchmarks in sequence.

    Usage:
        suite_runner = SuiteRunner(config, model, trainer)
        all_results = suite_runner.run_all_experiments(loaders)
    """

    def __init__(
        self,
        config,
        model: torch.nn.Module,
        trainer,
    ):
        """Initialize suite runner.

        Args:
            config: BenchmarkSuiteConfig
            model: Model to train
            trainer: Trainer instance
        """
        self.config = config
        self.model = model
        self.trainer = trainer

    def run_experiment(
        self,
        experiment_name: str,
        train_loader,
        val_loader=None,
    ) -> dict:
        """Run single experiment.

        Args:
            experiment_name: Name of experiment to run
            train_loader: Training data loader
            val_loader: Optional validation data loader

        Returns:
            Results dictionary
        """
        from geofm.experiments.adaptation_benchmark import AdaptationBenchmark

        exp_config = self.config.get_experiment_config(experiment_name)
        benchmark = AdaptationBenchmark(exp_config, self.model, self.trainer)

        runner = BenchmarkRunner(benchmark)
        return runner.run(train_loader, val_loader)

    def run_all_experiments(
        self,
        train_loader,
        val_loader=None,
    ) -> Dict[str, dict]:
        """Run all experiments in the suite.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader

        Returns:
            Dictionary mapping experiment name to results
        """
        results = {}

        for experiment_name in self.config.experiments:
            print(f"\nRunning experiment: {experiment_name}")

            results[experiment_name] = self.run_experiment(
                experiment_name,
                train_loader,
                val_loader,
            )

        return results