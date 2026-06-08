"""tests/test_csv_logger.py

Tests for CSVLogger.
"""
import pytest
from pathlib import Path

from geofm.logging.csv_logger import CSVLogger


class TestCSVLogger:
    """Test CSVLogger class."""

    def test_log_creates_file(self, tmp_path):
        """Logging should create file."""
        log_file = tmp_path / "metrics.csv"
        logger = CSVLogger(str(log_file))

        logger.log({"epoch": 1, "loss": 0.5})

        assert log_file.exists()

    def test_log_writes_data(self, tmp_path):
        """Logging should write data."""
        log_file = tmp_path / "metrics.csv"
        logger = CSVLogger(str(log_file))

        logger.log({"epoch": 1, "loss": 0.5})

        content = log_file.read_text()

        assert "epoch" in content
        assert "loss" in content
        assert "1" in content
        assert "0.5" in content

    def test_log_multiple_rows(self, tmp_path):
        """Multiple logs should create multiple rows."""
        log_file = tmp_path / "metrics.csv"
        logger = CSVLogger(str(log_file))

        logger.log({"epoch": 1, "loss": 0.5})
        logger.log({"epoch": 2, "loss": 0.4})
        logger.log({"epoch": 3, "loss": 0.3})

        content = log_file.read_text()
        lines = content.strip().split("\n")

        # Header + 3 data rows
        assert len(lines) == 4

    def test_log_header_once(self, tmp_path):
        """Header should only be written once."""
        log_file = tmp_path / "metrics.csv"
        logger = CSVLogger(str(log_file))

        logger.log({"epoch": 1, "loss": 0.5})
        logger.log({"epoch": 2, "loss": 0.4})

        content = log_file.read_text()
        lines = content.strip().split("\n")

        # Only one header line
        header_count = sum(1 for line in lines if "epoch" in line)
        assert header_count == 1

    def test_context_manager(self, tmp_path):
        """Should work as context manager."""
        log_file = tmp_path / "metrics.csv"

        with CSVLogger(str(log_file)) as logger:
            logger.log({"epoch": 1, "loss": 0.5})

        assert log_file.exists()

    def test_flush(self, tmp_path):
        """Flush should not error."""
        log_file = tmp_path / "metrics.csv"
        logger = CSVLogger(str(log_file))

        logger.log({"epoch": 1, "loss": 0.5})
        logger.flush()

        assert log_file.exists()

    def test_close(self, tmp_path):
        """Close should not error."""
        log_file = tmp_path / "metrics.csv"
        logger = CSVLogger(str(log_file))

        logger.log({"epoch": 1, "loss": 0.5})
        logger.close()

        assert log_file.exists()


class TestCSVLoggerEdgeCases:
    """Test edge cases."""

    def test_nested_path(self, tmp_path):
        """Should create nested directories."""
        log_file = tmp_path / "logs" / "metrics.csv"
        logger = CSVLogger(str(log_file))

        logger.log({"epoch": 1, "loss": 0.5})

        assert log_file.exists()

    def test_float_values(self, tmp_path):
        """Should handle float values."""
        log_file = tmp_path / "metrics.csv"
        logger = CSVLogger(str(log_file))

        logger.log({"epoch": 1, "loss": 0.123456789})

        content = log_file.read_text()
        assert "0.123" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])