"""geofm.logging.result_table

Result table for tracking experiment results and generating benchmark tables.
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Optional


class ResultTable:
    """Table for storing and managing experiment results.

    Usage:
        table = ResultTable()

        table.add_result(
            experiment="exp01",
            task="flood",
            adapter="lora",
            metrics={"iou": 0.85, "dice": 0.92},
        )

        table.save_csv("results/flood_benchmark.csv")

        # Load later
        loaded = ResultTable.load_csv("results/flood_benchmark.csv")
    """

    def __init__(self):
        """Initialize empty result table."""
        self.rows = []

    def add_result(
        self,
        experiment: str,
        task: str,
        adapter: str,
        metrics: dict,
    ):
        """Add a result row.

        Args:
            experiment: Experiment name (e.g., "exp01", "exp_lora_v1")
            task: Task name (e.g., "flood", "burn", "lulc")
            adapter: Adapter type (e.g., "lora", "feature", "hybrid")
            metrics: Dictionary of metric values
        """
        row = {
            "experiment": experiment,
            "task": task,
            "adapter": adapter,
        }

        # Add all metrics
        row.update(metrics)

        self.rows.append(row)

    def save_csv(
        self,
        path: str,
        index: bool = False,
    ):
        """Save results to CSV file.

        Args:
            path: Output CSV path
            index: Whether to write row indices
        """
        df = pd.DataFrame(self.rows)

        # Create directory if needed
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        if len(df) == 0:
            # Create empty file with headers
            df = pd.DataFrame(columns=["experiment", "task", "adapter"])

        df.to_csv(path, index=index)

    @classmethod
    def load_csv(
        cls,
        path: str,
    ) -> "ResultTable":
        """Load results from CSV file.

        Args:
            path: Input CSV path

        Returns:
            ResultTable with loaded data
        """
        table = cls()

        df = pd.read_csv(path)
        table.rows = df.to_dict("records")

        return table

    def get_pivot_table(
        self,
        index: str = "adapter",
        columns: str = "task",
        values: str = "iou",
    ) -> pd.DataFrame:
        """Create pivot table from results.

        Args:
            index: Column to use as index
            columns: Column to use as columns
            values: Metric column to aggregate

        Returns:
            Pivot table as DataFrame
        """
        df = pd.DataFrame(self.rows)
        return df.pivot_table(
            index=index,
            columns=columns,
            values=values,
        )

    def get_summary(
        self,
        group_by: str = "adapter",
    ) -> pd.DataFrame:
        """Get summary statistics grouped by a column.

        Args:
            group_by: Column to group by

        Returns:
            Summary statistics DataFrame
        """
        df = pd.DataFrame(self.rows)
        return df.groupby(group_by).agg({
            "iou": ["mean", "std", "max"],
            "dice": ["mean", "std", "max"],
            "f1": ["mean", "std", "max"],
        })

    def __len__(self):
        return len(self.rows)

    def __repr__(self):
        return f"ResultTable(rows={len(self.rows)})"