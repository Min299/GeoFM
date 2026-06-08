"""tests/test_logger.py

Tests for logger.
"""
import pytest
from pathlib import Path


class TestLogger:
    """Test Logger class."""

    def test_logger_init(self, tmp_path):
        """Logger should initialize."""
        from geofm.logging.logger import Logger

        logger = Logger(log_dir=tmp_path, verbose=False)

        assert logger.log_dir == tmp_path
        assert logger.file is not None

        logger.close()

    def test_log(self, tmp_path):
        """log should write message."""
        from geofm.logging.logger import Logger

        logger = Logger(log_dir=tmp_path, verbose=False)
        logger.log("test message")

        logger.close()

        # Check file
        log_files = list(tmp_path.glob("*.log"))
        assert len(log_files) == 1

        content = log_files[0].read_text()
        assert "test message" in content

    def test_info(self, tmp_path):
        """info should log with INFO level."""
        from geofm.logging.logger import Logger

        logger = Logger(log_dir=tmp_path, verbose=False)
        logger.info("info message")

        logger.close()

        content = list(tmp_path.glob("*.log"))[0].read_text()
        assert "INFO" in content

    def test_warning(self, tmp_path):
        """warning should log with WARNING level."""
        from geofm.logging.logger import Logger

        logger = Logger(log_dir=tmp_path, verbose=False)
        logger.warning("warn message")

        logger.close()

        content = list(tmp_path.glob("*.log"))[0].read_text()
        assert "WARNING" in content

    def test_context_manager(self, tmp_path):
        """Logger should work as context manager."""
        from geofm.logging.logger import Logger

        with Logger(log_dir=tmp_path, verbose=False) as logger:
            logger.log("test")

        # File should be closed


class TestGetLogger:
    """Test get_logger function."""

    def test_get_logger(self, tmp_path):
        """get_logger should return logger with prefix."""
        from geofm.logging.logger import get_logger

        logger = get_logger("test", log_dir=tmp_path)

        assert logger.prefix == "[test] "

        logger.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])