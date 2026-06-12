"""geofm/experiments.result_manager

Manage experiment result directories.
"""
from __future__ import annotations

from pathlib import Path


class ResultManager:
    """Manages result directories for experiment runs.

    Usage:
        manager = ResultManager("results")

        run_dir = manager.create_run_dir("exp_001")
        # Creates: results/exp_001/{checkpoints,metrics,predictions}/

        latest = manager.latest_run()
    """

    def __init__(
        self,
        root_dir: str = "results",
    ):
        """Initialize result manager.

        Args:
            root_dir: Root directory for all results
        """
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def create_run_dir(
        self,
        run_name: str,
    ) -> Path:
        """Create a directory structure for a run.

        Creates:
            run_dir/
            ├── checkpoints/
            ├── metrics/
            └── predictions/

        Args:
            run_name: Name of the run

        Returns:
            Path to run directory
        """
        run_dir = self.root_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        (run_dir / "metrics").mkdir(exist_ok=True)
        (run_dir / "predictions").mkdir(exist_ok=True)

        return run_dir

    def latest_run(self) -> Path | None:
        """Get the most recent run directory.

        Returns:
            Path to latest run or None if no runs exist
        """
        runs = sorted(
            self.root_dir.iterdir(),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if len(runs) == 0:
            return None

        return runs[0]

    def list_runs(self) -> list[Path]:
        """List all run directories.

        Returns:
            List of run directory paths sorted by creation time
        """
        runs = sorted(
            self.root_dir.iterdir(),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return runs

    def get_run_dir(self, run_name: str) -> Path:
        """Get path to a specific run directory.

        Args:
            run_name: Name of the run

        Returns:
            Path to run directory
        """
        return self.root_dir / run_name