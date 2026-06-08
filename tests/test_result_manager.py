"""tests/test_result_manager.py

Tests for ResultManager.
"""
import pytest
from pathlib import Path

from geofm.experiments.result_manager import ResultManager


class TestResultManager:
    """Test ResultManager class."""

    def test_create_run_dir(self, tmp_path):
        """Creating run dir should create structure."""
        manager = ResultManager(tmp_path)

        run_dir = manager.create_run_dir("exp_001")

        assert run_dir.exists()
        assert (run_dir / "checkpoints").exists()
        assert (run_dir / "metrics").exists()
        assert (run_dir / "predictions").exists()

    def test_create_run_dir_nested(self, tmp_path):
        """Should create parent directories."""
        manager = ResultManager(tmp_path)

        run_dir = manager.create_run_dir("nested/exp_001")

        assert run_dir.exists()

    def test_latest_run_none(self, tmp_path):
        """Latest run should be None when empty."""
        manager = ResultManager(tmp_path)

        latest = manager.latest_run()

        assert latest is None

    def test_latest_run(self, tmp_path):
        """Latest run should be most recent."""
        import time

        manager = ResultManager(tmp_path)

        # Create runs with delay to ensure different mtime
        manager.create_run_dir("exp_001")
        time.sleep(0.01)
        manager.create_run_dir("exp_002")
        time.sleep(0.01)
        manager.create_run_dir("exp_003")

        latest = manager.latest_run()

        assert latest.name == "exp_003"

    def test_list_runs(self, tmp_path):
        """List runs should return all runs sorted."""
        manager = ResultManager(tmp_path)

        manager.create_run_dir("exp_001")
        manager.create_run_dir("exp_002")

        runs = manager.list_runs()

        assert len(runs) == 2

    def test_get_run_dir(self, tmp_path):
        """Get run dir should return path."""
        manager = ResultManager(tmp_path)

        manager.create_run_dir("exp_001")

        run_dir = manager.get_run_dir("exp_001")

        assert run_dir.exists()
        assert run_dir.name == "exp_001"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])