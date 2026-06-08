"""geofm.evaluation.results_writer

Writers for saving benchmark results to disk.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime


class ResultsWriter:
    """Writer for saving benchmark results.

    Saves results to JSON files in a structured directory layout.

    Usage:
        writer = ResultsWriter(output_dir="results")
        writer.write("feature", metrics)
    """

    def __init__(
        self,
        output_dir: str = "results",
        timestamp: bool = True,
    ):
        """Initialize results writer.

        Args:
            output_dir: Base directory for results
            timestamp: If True, add timestamp to directory
        """
        self.output_dir = Path(output_dir)

        if timestamp:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = self.output_dir / timestamp_str

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        experiment_name: str,
        metrics: Dict,
        tag: Optional[str] = None,
    ) -> Path:
        """Write results for an experiment.

        Args:
            experiment_name: Name of experiment
            metrics: Metrics dictionary
            tag: Optional tag (e.g., "final", "checkpoint")

        Returns:
            Path to written file
        """
        exp_dir = self.output_dir / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        if tag:
            filename = f"{tag}.json"
        else:
            filename = "summary.json"

        filepath = exp_dir / filename

        with open(filepath, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        return filepath

    def write_history(
        self,
        experiment_name: str,
        history: list,
    ) -> Path:
        """Write training history.

        Args:
            experiment_name: Name of experiment
            history: List of epoch metrics

        Returns:
            Path to written file
        """
        exp_dir = self.output_dir / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        filepath = exp_dir / "history.json"

        with open(filepath, "w") as f:
            json.dump(history, f, indent=2, default=str)

        return filepath

    def write_checkpoint(
        self,
        experiment_name: str,
        epoch: int,
        state: Dict,
    ) -> Path:
        """Write checkpoint data.

        Args:
            experiment_name: Name of experiment
            epoch: Epoch number
            state: State dictionary

        Returns:
            Path to written file
        """
        exp_dir = self.output_dir / experiment_name / "checkpoints"
        exp_dir.mkdir(parents=True, exist_ok=True)

        filepath = exp_dir / f"epoch_{epoch:04d}.json"

        with open(filepath, "w") as f:
            json.dump(state, f, indent=2, default=str)

        return filepath

    def write_summary(
        self,
        all_results: Dict[str, Dict],
    ) -> Path:
        """Write summary of all experiments.

        Args:
            all_results: Dictionary mapping experiment name to metrics

        Returns:
            Path to written file
        """
        filepath = self.output_dir / "summary.json"

        summary = {
            "timestamp": datetime.now().isoformat(),
            "experiments": list(all_results.keys()),
            "results": all_results,
        }

        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        return filepath

    def get_output_dir(self) -> Path:
        """Get output directory.

        Returns:
            Output directory path
        """
        return self.output_dir


class CSVResultsWriter:
    """Writer for CSV format results."""

    def __init__(self, output_dir: str = "results"):
        """Initialize CSV writer.

        Args:
            output_dir: Base directory for results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_row(self, experiment_name: str, metrics: Dict) -> Path:
        """Append a row to CSV file.

        Args:
            experiment_name: Name of experiment
            metrics: Metrics dictionary

        Returns:
            Path to written file
        """
        import csv

        filepath = self.output_dir / f"{experiment_name}.csv"
        file_exists = filepath.exists()

        with open(filepath, "a", newline="") as f:
            if not file_exists:
                writer = csv.DictWriter(f, fieldnames=metrics.keys())
                writer.writeheader()
            else:
                writer = csv.DictWriter(f, fieldnames=metrics.keys())

            writer.writerow(metrics)

        return filepath


class TensorBoardWriter:
    """Writer for TensorBoard logs.

    Requires tensorboard package.
    """

    def __init__(self, log_dir: str = "runs"):
        """Initialize TensorBoard writer.

        Args:
            log_dir: Directory for TensorBoard logs
        """
        self.log_dir = log_dir
        self.writer = None

    def __enter__(self):
        """Context manager entry."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.log_dir)
        except ImportError:
            print("Warning: tensorboard not available")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.writer:
            self.writer.close()

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log scalar value.

        Args:
            tag: Tag name
            value: Scalar value
            step: Step number
        """
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag: str, values: Dict[str, float], step: int) -> None:
        """Log multiple scalars.

        Args:
            tag: Tag name
            values: Dictionary of scalar values
            step: Step number
        """
        if self.writer:
            self.writer.add_scalars(tag, values, step)