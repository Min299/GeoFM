"""tests/test_experiment_tracker.py

Tests for ExperimentTracker.
"""
import pytest
from pathlib import Path

from geofm.experiments.experiment_tracker import (
    ExperimentTracker,
    ExperimentRecord,
)


class TestExperimentTracker:
    """Test ExperimentTracker class."""

    def test_start_run(self, tmp_path):
        """Starting a run should create a record."""
        tracker = ExperimentTracker(tmp_path)

        run_id = tracker.start_run(
            experiment_name="exp01",
            task="flood",
            adapter_type="lora",
        )

        assert run_id is not None
        assert tracker.active_run is not None
        assert tracker.active_run.experiment_name == "exp01"
        assert tracker.active_run.task == "flood"
        assert tracker.active_run.adapter_type == "lora"

    def test_log_metric(self, tmp_path):
        """Logging metrics should append to list."""
        tracker = ExperimentTracker(tmp_path)

        tracker.start_run("exp01", "flood", "lora")

        tracker.log_metric("loss", 0.5)
        tracker.log_metric("iou", 0.85)

        assert len(tracker.active_run.metrics["loss"]) == 1
        assert tracker.active_run.metrics["loss"][0] == 0.5
        assert len(tracker.active_run.metrics["iou"]) == 1
        assert tracker.active_run.metrics["iou"][0] == 0.85

    def test_log_metric_multiple(self, tmp_path):
        """Multiple metrics with same name should accumulate."""
        tracker = ExperimentTracker(tmp_path)

        tracker.start_run("exp01", "flood", "lora")

        tracker.log_metric("loss", 0.5)
        tracker.log_metric("loss", 0.4)
        tracker.log_metric("loss", 0.3)

        assert len(tracker.active_run.metrics["loss"]) == 3
        assert tracker.active_run.metrics["loss"] == [0.5, 0.4, 0.3]

    def test_log_checkpoint(self, tmp_path):
        """Logging checkpoint should add to list."""
        tracker = ExperimentTracker(tmp_path)

        tracker.start_run("exp01", "flood", "lora")

        tracker.log_checkpoint("checkpoints/model.pt")
        tracker.log_checkpoint("checkpoints/model_best.pt")

        assert len(tracker.active_run.checkpoints) == 2
        assert "model.pt" in tracker.active_run.checkpoints[0]

    def test_finish_run(self, tmp_path):
        """Finishing run should set end time."""
        tracker = ExperimentTracker(tmp_path)

        tracker.start_run("exp01", "flood", "lora")
        tracker.log_metric("loss", 0.5)

        record = tracker.finish_run()

        assert record.end_time is not None
        assert tracker.active_run is None

    def test_no_active_run_error(self, tmp_path):
        """Logging without active run should raise error."""
        tracker = ExperimentTracker(tmp_path)

        with pytest.raises(RuntimeError, match="No active experiment"):
            tracker.log_metric("loss", 0.5)

    def test_is_active(self, tmp_path):
        """is_active should reflect state."""
        tracker = ExperimentTracker(tmp_path)

        assert not tracker.is_active()

        tracker.start_run("exp01", "flood", "lora")

        assert tracker.is_active()

        tracker.finish_run()

        assert not tracker.is_active()


class TestExperimentRecord:
    """Test ExperimentRecord dataclass."""

    def test_create_record(self):
        """Creating record with defaults."""
        record = ExperimentRecord(
            experiment_name="exp01",
            task="flood",
            adapter_type="lora",
            run_id="20240101_120000",
            start_time="2024-01-01 12:00:00",
        )

        assert record.end_time is None
        assert record.metrics == {}
        assert record.checkpoints == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])