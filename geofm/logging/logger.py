"""geofm.logging.logger

Basic file logger for experiments.

Provides simple file-based logging.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Optional, Union
import sys


class Logger:
    """Simple file logger for experiments.

    Logs messages to both file and optionally stdout.

    Usage:
        logger = Logger("logs/my_experiment")
        logger.log("Starting training...")
        logger.log("Epoch 1 complete")
        logger.close()
    """

    def __init__(
        self,
        log_dir: Union[str, Path] = "logs",
        filename: Optional[str] = None,
        prefix: str = "",
        verbose: bool = True,
    ):
        """Initialize logger.

        Args:
            log_dir: Directory for log files
            filename: Optional custom filename (default: timestamp)
            prefix: Optional prefix for all messages
            verbose: Whether to also print to stdout
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.verbose = verbose

        # Generate filename
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}.log"

        self.filepath = self.log_dir / filename
        self.file = open(self.filepath, "a")

    def log(
        self,
        message: str,
        level: str = "INFO",
    ) -> None:
        """Log a message.

        Args:
            message: Message to log
            level: Log level (INFO, WARNING, ERROR)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] [{level}] {self.prefix}{message}"

        self.file.write(formatted + "\n")
        self.file.flush()

        if self.verbose:
            print(formatted)

    def info(self, message: str) -> None:
        """Log info message.

        Args:
            message: Message to log
        """
        self.log(message, level="INFO")

    def warning(self, message: str) -> None:
        """Log warning message.

        Args:
            message: Message to log
        """
        self.log(message, level="WARNING")

    def error(self, message: str) -> None:
        """Log error message.

        Args:
            message: Message to log
        """
        self.log(message, level="ERROR")

    def debug(self, message: str) -> None:
        """Log debug message.

        Args:
            message: Message to log
        """
        self.log(message, level="DEBUG")

    def section(self, title: str) -> None:
        """Log a section header.

        Args:
            title: Section title
        """
        separator = "=" * 60
        self.log(separator)
        self.log(title)
        self.log(separator)

    def subsection(self, title: str) -> None:
        """Log a subsection header.

        Args:
            title: Subsection title
        """
        separator = "-" * 40
        self.log(separator)
        self.log(title)

    def close(self) -> None:
        """Close the log file."""
        if hasattr(self, "file") and self.file:
            self.file.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Destructor to ensure file is closed."""
        self.close()


class MultiLogger:
    """Logger that writes to multiple outputs.

    Usage:
        logger = MultiLogger([
            Logger("logs/experiments"),
            Logger("logs/debug"),
        ])
        logger.log("Test message")
    """

    def __init__(self, loggers: list):
        """Initialize multi-logger.

        Args:
            loggers: List of Logger instances
        """
        self.loggers = loggers

    def log(self, message: str, level: str = "INFO") -> None:
        """Log to all loggers.

        Args:
            message: Message to log
            level: Log level
        """
        for logger in self.loggers:
            logger.log(message, level)

    def close(self) -> None:
        """Close all loggers."""
        for logger in self.loggers:
            logger.close()


def get_logger(
    name: str,
    log_dir: str = "logs",
) -> Logger:
    """Get a logger with the given name.

    Args:
        name: Logger name (used as prefix)
        log_dir: Log directory

    Returns:
        Logger instance
    """
    return Logger(log_dir=log_dir, prefix=f"[{name}] ")