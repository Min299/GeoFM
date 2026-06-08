"""geofm.experiments.multitask_experiment

Multi-task experiment runner.

Provides experiment abstraction for multi-task learning.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import time

from geofm.training.multitask_trainer import (
    MultiTaskTrainer,
    EpochMetrics,
)
from geofm.tasks.task_batcher import TaskBatcher
from geofm.evaluation.multitask_metrics import MultiTaskMetrics


@dataclass
class ExperimentConfig:
    """Configuration for a multi-task experiment."""

    name: str
    tasks: List[str]
    epochs: int = 10
    steps_per_epoch: int = 100
    adapter_type: str = "feature"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    clip_grad: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "name": self.name,
            "tasks": self.tasks,
            "epochs": self.epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "adapter_type": self.adapter_type,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "clip_grad": self.clip_grad,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ExperimentConfig":
        """Create from dictionary.

        Args:
            data: Dictionary with config data

        Returns:
            ExperimentConfig instance
        """
        return cls(**data)


@dataclass
class ExperimentResult:
    """Result from a multi-task experiment."""

    config: ExperimentConfig
    history: List[Dict] = field(default_factory=list)
    final_metrics: Dict[str, float] = field(default_factory=dict)
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "config": self.config.to_dict(),
            "history": self.history,
            "final_metrics": self.final_metrics,
            "duration_seconds": self.duration_seconds,
        }


class MultiTaskExperiment:
    """Multi-task experiment runner.

    Provides a complete experiment pipeline for multi-task learning.

    Usage:
        exp = MultiTaskExperiment(
            trainer=trainer,
            config=ExperimentConfig(
                name="feature",
                tasks=["flood", "burn", "lulc"],
                epochs=10,
            ),
        )
        result = exp.run(task_batcher)
    """

    def __init__(
        self,
        trainer: MultiTaskTrainer,
        config: ExperimentConfig,
        metrics_tracker: Optional[MultiTaskMetrics] = None,
    ):
        """Initialize multi-task experiment.

        Args:
            trainer: MultiTaskTrainer instance
            config: ExperimentConfig
            metrics_tracker: Optional metrics tracker
        """
        self.trainer = trainer
        self.config = config
        self.metrics_tracker = metrics_tracker or MultiTaskMetrics()
        self.history: List[Dict] = []

    def run(
        self,
        task_batcher: TaskBatcher,
        eval_batcher: Optional[TaskBatcher] = None,
    ) -> ExperimentResult:
        """Run the experiment.

        Args:
            task_batcher: TaskBatcher for training
            eval_batcher: Optional TaskBatcher for evaluation

        Returns:
            ExperimentResult with experiment results
        """
        start_time = time.time()
        self.history = []

        for epoch in range(self.config.epochs):
            # Train epoch
            train_metrics = self.trainer.train_epoch(
                task_batcher,
                num_steps=self.config.steps_per_epoch,
            )

            # Record training metrics
            epoch_data = {
                "epoch": epoch,
                "train_loss": train_metrics.avg_loss,
                "train_steps": train_metrics.total_steps,
            }

            # Evaluate if eval_batcher provided
            if eval_batcher is not None:
                eval_metrics = self.trainer.evaluate(
                    eval_batcher,
                    num_steps=self.config.steps_per_epoch,
                )
                epoch_data["eval_loss"] = eval_metrics.avg_loss

            self.history.append(epoch_data)

            # Update metrics tracker
            for task, losses in train_metrics.task_losses.items():
                for loss in losses:
                    self.metrics_tracker.update(task, loss=loss)

        duration = time.time() - start_time

        # Compute final metrics
        final_metrics = self._compute_final_metrics()

        return ExperimentResult(
            config=self.config,
            history=self.history,
            final_metrics=final_metrics,
            duration_seconds=duration,
        )

    def _compute_final_metrics(self) -> Dict[str, float]:
        """Compute final metrics.

        Returns:
            Dictionary with final metrics
        """
        summary = self.metrics_tracker.summary()

        metrics = {}
        for task, task_summary in summary.items():
            metrics[f"{task}_final_loss"] = task_summary.get("final_loss", 0)
            metrics[f"{task}_best_loss"] = task_summary.get("best_loss", 0)
            metrics[f"{task}_avg_loss"] = task_summary.get("avg_loss", 0)

        return metrics

    def get_history(self) -> List[Dict]:
        """Get training history.

        Returns:
            List of epoch data dictionaries
        """
        return self.history

    def get_best_epoch(self, metric: str = "train_loss") -> int:
        """Get epoch with best metric.

        Args:
            metric: Metric to optimize

        Returns:
            Best epoch number
        """
        if not self.history:
            return 0

        values = [epoch.get(metric, float("inf")) for epoch in self.history]
        return values.index(min(values))


class MultiTaskExperimentSuite:
    """Suite of multi-task experiments.

    Runs multiple experiments and compares results.

    Usage:
        suite = MultiTaskExperimentSuite()
        suite.add_experiment("feature", trainer, config)
        suite.add_experiment("lora", trainer, config)
        results = suite.run_all(task_batcher)
    """

    def __init__(self):
        """Initialize experiment suite."""
        self.experiments: Dict[str, MultiTaskExperiment] = {}
        self.results: Dict[str, ExperimentResult] = {}

    def add_experiment(
        self,
        name: str,
        trainer: MultiTaskTrainer,
        config: ExperimentConfig,
    ) -> None:
        """Add an experiment to the suite.

        Args:
            name: Experiment name
            trainer: MultiTaskTrainer
            config: ExperimentConfig
        """
        self.experiments[name] = MultiTaskExperiment(
            trainer=trainer,
            config=config,
        )

    def run_all(
        self,
        task_batcher: TaskBatcher,
    ) -> Dict[str, ExperimentResult]:
        """Run all experiments.

        Args:
            task_batcher: TaskBatcher for training

        Returns:
            Dictionary mapping experiment name to results
        """
        for name, experiment in self.experiments.items():
            print(f"Running experiment: {name}")
            self.results[name] = experiment.run(task_batcher)

        return self.results

    def get_results(self) -> Dict[str, ExperimentResult]:
        """Get all experiment results.

        Returns:
            Dictionary of results
        """
        return self.results

    def get_best_experiment(
        self,
        task: str,
        metric: str = "final_loss",
    ) -> Optional[str]:
        """Get best experiment for a task.

        Args:
            task: Task name
            metric: Metric to optimize

        Returns:
            Best experiment name or None
        """
        scores = {}

        for name, result in self.results.items():
            key = f"{task}_{metric}"
            if key in result.final_metrics:
                scores[name] = result.final_metrics[key]

        if not scores:
            return None

        return min(scores, key=scores.get)

    def to_dict(self) -> Dict:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "experiments": list(self.experiments.keys()),
            "results": {
                name: result.to_dict()
                for name, result in self.results.items()
            },
        }