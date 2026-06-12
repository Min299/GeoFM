"""geofm.logging.experiment_logger

Experiment logger combining multiple logging utilities.

Provides a unified interface for experiment logging.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Union
import json

from geofm.logging.metric_logger import MetricLogger, TaskMetricLogger
from geofm.logging.logger import Logger


class ExperimentLogger:
    """Combined logger for experiments.

    Provides metric tracking, file logging, and experiment management.

    Usage:
        logger = ExperimentLogger("flood_experiment")

        logger.log_metric("loss", 0.5)
        logger.log_metric("accuracy", 0.8)

        logger.info("Starting training...")
        logger.save()
    """

    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "logs",
        output_dir: str = "outputs",
        use_task_metrics: bool = False,
    ):
        """Initialize experiment logger.

        Args:
            experiment_name: Name of the experiment
            log_dir: Directory for log files
            output_dir: Directory for outputs
            use_task_metrics: Whether to use task-specific metrics
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.output_dir = Path(output_dir)

        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize logger and metrics
        self.logger = Logger(
            log_dir=self.log_dir,
            prefix=f"[{experiment_name}] ",
        )

        if use_task_metrics:
            self.metrics = TaskMetricLogger()
        else:
            self.metrics = MetricLogger()

        self.start_step = 0
        self.epoch = 0

    def log_metric(
        self,
        key: str,
        value: float,
        step: Optional[int] = None,
    ) -> None:
        """Log a metric value.

        Args:
            key: Metric name
            value: Metric value
            step: Optional step number
        """
        self.metrics.update(key, value, step)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log multiple metrics.

        Args:
            metrics: Dictionary of metrics
            step: Optional step number
        """
        for key, value in metrics.items():
            self.log_metric(key, value, step)

    def log_task_metric(
        self,
        task: str,
        key: str,
        value: float,
        step: Optional[int] = None,
    ) -> None:
        """Log a metric for a specific task.

        Args:
            task: Task name
            key: Metric name
            value: Metric value
            step: Optional step number
        """
        if isinstance(self.metrics, TaskMetricLogger):
            self.metrics.update(key, value, task=task, step=step)

    def info(self, message: str) -> None:
        """Log info message.

        Args:
            message: Message to log
        """
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log warning message.

        Args:
            message: Message to log
        """
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message.

        Args:
            message: Message to log
        """
        self.logger.error(message)

    def log_epoch(
        self,
        epoch: int,
        metrics: Dict[str, float],
    ) -> None:
        """Log metrics for an epoch.

        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics
        """
        self.epoch = epoch
        self.info(f"Epoch {epoch}")

        for key, value in metrics.items():
            self.log_metric(f"epoch_{key}", value)
            self.logger.log(f"  {key}: {value:.4f}", level="INFO")

    def start_epoch(self, epoch: int) -> None:
        """Mark the start of an epoch.

        Args:
            epoch: Epoch number
        """
        self.epoch = epoch
        self.logger.subsection(f"Epoch {epoch}")

    def end_epoch(self, metrics: Optional[Dict[str, float]] = None) -> None:
        """Mark the end of an epoch.

        Args:
            metrics: Optional epoch metrics
        """
        if metrics:
            self.log_epoch(self.epoch, metrics)

    def summary(self) -> Dict[str, Any]:
        """Get experiment summary.

        Returns:
            Dictionary with experiment info and metrics
        """
        return {
            "experiment_name": self.experiment_name,
            "epoch": self.epoch,
            "step": self.metrics.step,
            "metrics": self.metrics.summary(),
            "detailed_metrics": self.metrics.detailed_summary(),
        }

    def save(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        """Save experiment state to file.

        Args:
            filepath: Optional output path

        Returns:
            Path to saved file
        """
        if filepath is None:
            filepath = self.output_dir / f"{self.experiment_name}_metrics.json"
        else:
            filepath = Path(filepath)

        data = self.summary()

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        self.info(f"Saved metrics to {filepath}")

        return filepath

    def save_metrics_csv(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        """Save metrics history to CSV.

        Args:
            filepath: Optional output path

        Returns:
            Path to saved file
        """
        if filepath is None:
            filepath = self.output_dir / f"{self.experiment_name}_metrics.csv"
        else:
            filepath = Path(filepath)

        import csv

        # Get all keys
        all_keys = set()
        for key in self.metrics.history:
            all_keys.update(self.metrics.history.keys())

        # Write CSV
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "metric", "value"])

            for key, values in self.metrics.history.items():
                for step, value in enumerate(values):
                    writer.writerow([step, key, value])

        return filepath

    def close(self) -> None:
        """Close the logger."""
        self.logger.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def create_experiment_logger(
    experiment_name: str,
    log_dir: str = "logs",
    output_dir: str = "outputs",
) -> ExperimentLogger:
    """Create an experiment logger.

    Args:
        experiment_name: Name of the experiment
        log_dir: Directory for log files
        output_dir: Directory for outputs

    Returns:
        ExperimentLogger instance
    """
    return ExperimentLogger(
        experiment_name=experiment_name,
        log_dir=log_dir,
        output_dir=output_dir,
    )