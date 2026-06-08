"""tests/test_metric_logger.py

Tests for metric logger.
"""
import pytest


class TestMetricLogger:
    """Test MetricLogger class."""

    def test_init(self):
        """MetricLogger should initialize."""
        from geofm.logging.metric_logger import MetricLogger

        m = MetricLogger()

        assert len(m.history) == 0

    def test_update(self):
        """update should add value."""
        from geofm.logging.metric_logger import MetricLogger

        m = MetricLogger()

        m.update("loss", 1.0)
        m.update("loss", 0.5)

        assert len(m.history["loss"]) == 2

    def test_latest(self):
        """latest should return last value."""
        from geofm.logging.metric_logger import MetricLogger

        m = MetricLogger()

        m.update("loss", 1.0)
        m.update("loss", 0.5)

        assert m.latest("loss") == 0.5

    def test_latest_nonexistent(self):
        """latest should return None for nonexistent."""
        from geofm.logging.metric_logger import MetricLogger

        m = MetricLogger()

        assert m.latest("nonexistent") is None

    def test_mean(self):
        """mean should compute average."""
        from geofm.logging.metric_logger import MetricLogger

        m = MetricLogger()

        m.update("loss", 1.0)
        m.update("loss", 2.0)

        assert abs(m.mean("loss") - 1.5) < 0.01

    def test_summary(self):
        """summary should return mean for all metrics."""
        from geofm.logging.metric_logger import MetricLogger

        m = MetricLogger()

        m.update("loss", 1.0)
        m.update("loss", 0.5)
        m.update("accuracy", 0.8)

        summary = m.summary()

        assert "loss" in summary
        assert "accuracy" in summary

    def test_clear(self):
        """clear should reset all metrics."""
        from geofm.logging.metric_logger import MetricLogger

        m = MetricLogger()

        m.update("loss", 1.0)
        m.clear()

        assert len(m.history) == 0

    def test_has_metric(self):
        """has_metric should check existence."""
        from geofm.logging.metric_logger import MetricLogger

        m = MetricLogger()

        m.update("loss", 1.0)

        assert m.has_metric("loss")
        assert not m.has_metric("accuracy")

    def test_get_recent(self):
        """get_recent should return last N values."""
        from geofm.logging.metric_logger import MetricLogger

        m = MetricLogger()

        for i in range(10):
            m.update("loss", float(i))

        recent = m.get_recent("loss", n=3)

        assert len(recent) == 3
        assert recent[-1] == 9.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])