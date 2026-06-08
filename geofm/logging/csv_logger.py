"""geofm.logging.csv_logger

CSV logger for experiment metrics.
"""
from __future__ import annotations

from pathlib import Path
import csv


class CSVLogger:
    """Logger that writes metrics to CSV file.

    Usage:
        logger = CSVLogger("metrics/train.csv")

        logger.log({"epoch": 1, "loss": 0.5, "iou": 0.85})
        logger.log({"epoch": 2, "loss": 0.4, "iou": 0.87})

        logger.flush()
    """

    def __init__(
        self,
        output_file: str,
    ):
        """Initialize CSV logger.

        Args:
            output_file: Path to output CSV file
        """
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.headers_written = False

    def log(self, row: dict):
        """Log a row of metrics.

        Args:
            row: Dictionary of metric values
        """
        write_header = not self.output_file.exists()

        with open(
            self.output_file,
            "a",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=row.keys(),
            )

            if write_header:
                writer.writeheader()

            writer.writerow(row)

    def flush(self):
        """Flush any buffered writes."""
        pass

    def close(self):
        """Close the logger."""
        self.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()