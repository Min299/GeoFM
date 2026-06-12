"""geofm/experiments.experiment_tracker

Track experiment runs with metrics and checkpoints.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ExperimentRecord:
    """Record for a single experiment run."""

    experiment_name: str
    task: str
    adapter_type: str

    run_id: str

    start_time: str
    end_time: str | None = None

    metrics: dict[str, list[float]] = field(
        default_factory=dict
    )

    checkpoints: list[str] = field(
        default_factory=list
    )


class ExperimentTracker:
    """Tracker for experiment runs.

    Usage:
        tracker = ExperimentTracker("outputs/experiments")

        run_id = tracker.start_run(
            experiment_name="exp01",
            task="flood",
            adapter_type="lora",
        )

        tracker.log_metric("loss", 0.5)
        tracker.log_metric("iou", 0.85)

        tracker.log_checkpoint("checkpoints/model.pt")

        record = tracker.finish_run()
    """

    def __init__(
        self,
        output_dir: str | Path,
    ):
        """Initialize tracker.

        Args:
            output_dir: Directory to store experiment records
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.active_run: ExperimentRecord | None = None

    def start_run(
        self,
        experiment_name: str,
        task: str,
        adapter_type: str,
    ) -> str:
        """Start a new experiment run.

        Args:
            experiment_name: Name of the experiment
            task: Task type (flood, burn, lulc)
            adapter_type: Adapter type (lora, feature, hybrid)

        Returns:
            Unique run ID
        """
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.active_run = ExperimentRecord(
            experiment_name=experiment_name,
            task=task,
            adapter_type=adapter_type,
            run_id=run_id,
            start_time=str(datetime.now()),
        )

        return run_id

    def log_metric(
        self,
        metric_name: str,
        value: float,
    ):
        """Log a metric value.

        Args:
            metric_name: Name of the metric
            value: Metric value

        Raises:
            RuntimeError: If no active experiment run
        """
        if self.active_run is None:
            raise RuntimeError("No active experiment run.")

        self.active_run.metrics.setdefault(
            metric_name,
            [],
        ).append(float(value))

    def log_checkpoint(
        self,
        checkpoint_path: str,
    ):
        """Log a checkpoint path.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        if self.active_run is None:
            raise RuntimeError("No active experiment run.")

        self.active_run.checkpoints.append(checkpoint_path)

    def finish_run(self) -> ExperimentRecord:
        """Finish the current run.

        Returns:
            The experiment record
        """
        if self.active_run is None:
            raise RuntimeError("No active experiment run.")

        self.active_run.end_time = str(datetime.now())

        record = self.active_run
        self.active_run = None

        return record

    def is_active(self) -> bool:
        """Check if there's an active run.

        Returns:
            True if a run is active
        """
        return self.active_run is not None