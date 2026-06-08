"""tests/test_multitask_metrics.py

Tests for multi-task metrics.
"""
import pytest


class TestMultiTaskMetrics:
    """Test MultiTaskMetrics class."""

    def test_init(self):
        """MultiTaskMetrics should initialize."""
        from geofm.evaluation.multitask_metrics import MultiTaskMetrics

        metrics = MultiTaskMetrics()

        assert len(metrics.task_metrics) == 0

    def test_update(self):
        """update should add loss for task."""
        from geofm.evaluation.multitask_metrics import MultiTaskMetrics

        metrics = MultiTaskMetrics()

        metrics.update("flood", loss=0.5)

        assert "flood" in metrics.task_metrics

    def test_summary(self):
        """summary should return all task metrics."""
        from geofm.evaluation.multitask_metrics import MultiTaskMetrics

        metrics = MultiTaskMetrics()

        metrics.update("flood", loss=0.5)
        metrics.update("flood", loss=0.4)
        metrics.update("burn", loss=0.6)

        summary = metrics.summary()

        assert "flood" in summary
        assert "burn" in summary

    def test_get_avg_loss(self):
        """get_avg_loss should return average."""
        from geofm.evaluation.multitask_metrics import MultiTaskMetrics

        metrics = MultiTaskMetrics()

        metrics.update("flood", loss=0.5)
        metrics.update("flood", loss=0.7)

        avg = metrics.get_avg_loss("flood")

        assert abs(avg - 0.6) < 0.01

    def test_reset(self):
        """reset should clear all metrics."""
        from geofm.evaluation.multitask_metrics import MultiTaskMetrics

        metrics = MultiTaskMetrics()

        metrics.update("flood", loss=0.5)

        metrics.reset()

        assert len(metrics.task_metrics) == 0


class TestMetricTracker:
    """Test MetricTracker class."""

    def test_add(self):
        """add should store value."""
        from geofm.evaluation.multitask_metrics import MetricTracker

        tracker = MetricTracker()

        tracker.add(0.5)
        tracker.add(0.6)

        assert len(tracker.values) == 2

    def test_get_moving_avg(self):
        """get_moving_avg should return average."""
        from geofm.evaluation.multitask_metrics import MetricTracker

        tracker = MetricTracker(window_size=3)

        tracker.add(0.5)
        tracker.add(0.6)
        tracker.add(0.7)

        avg = tracker.get_moving_avg()

        assert abs(avg - 0.6) < 0.01

    def test_get_best(self):
        """get_best should return minimum."""
        from geofm.evaluation.multitask_metrics import MetricTracker

        tracker = MetricTracker()

        tracker.add(0.5)
        tracker.add(0.3)
        tracker.add(0.7)

        best = tracker.get_best()

        assert best == 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])