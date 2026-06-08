"""tests/test_metrics_logger.py

Tests for MetricsLogger.
"""
import pytest

from geofm.logging.metrics_logger import MetricsLogger


class TestMetricsLogger:
    """Test MetricsLogger class."""

    def test_empty_logger(self):
        """Empty logger should have no history."""
        logger = MetricsLogger()
        assert len(logger) == 0
        assert logger.latest() is None

    def test_log_single_entry(self):
        """Logging should add to history."""
        logger = MetricsLogger()

        logger.log(1, "train", {"loss": 0.5, "iou": 0.7})

        assert len(logger) == 1
        assert logger.latest()["epoch"] == 1
        assert logger.latest()["split"] == "train"
        assert logger.latest()["loss"] == 0.5

    def test_log_multiple_epochs(self):
        """Should track multiple epochs."""
        logger = MetricsLogger()

        for epoch in range(5):
            logger.log(epoch, "train", {"loss": 0.5 - epoch * 0.1})

        assert len(logger) == 5
        assert logger.latest()["epoch"] == 4

    def test_log_train_and_val(self):
        """Should track both train and val splits."""
        logger = MetricsLogger()

        logger.log(1, "train", {"loss": 0.5})
        logger.log(1, "val", {"loss": 0.6})
        logger.log(2, "train", {"loss": 0.4})

        assert len(logger) == 3

        # Latest should be train
        assert logger.latest()["split"] == "train"

        # Latest val should be epoch 1
        latest_val = logger.latest("val")
        assert latest_val["epoch"] == 1
        assert latest_val["split"] == "val"

    def test_get_history(self):
        """Should return full or filtered history."""
        logger = MetricsLogger()

        logger.log(1, "train", {"loss": 0.5})
        logger.log(1, "val", {"loss": 0.6})
        logger.log(2, "train", {"loss": 0.4})

        train_history = logger.get_history("train")
        assert len(train_history) == 2
        assert all(r["split"] == "train" for r in train_history)

        full_history = logger.get_history()
        assert len(full_history) == 3

    def test_get_best_max(self):
        """Should find best when maximizing."""
        logger = MetricsLogger()

        logger.log(1, "train", {"iou": 0.7})
        logger.log(2, "train", {"iou": 0.85})
        logger.log(3, "train", {"iou": 0.8})

        best = logger.get_best("iou", mode="max")

        assert best["epoch"] == 2
        assert best["iou"] == 0.85

    def test_get_best_min(self):
        """Should find best when minimizing."""
        logger = MetricsLogger()

        logger.log(1, "train", {"loss": 0.5})
        logger.log(2, "train", {"loss": 0.3})
        logger.log(3, "train", {"loss": 0.4})

        best = logger.get_best("loss", mode="min")

        assert best["epoch"] == 2
        assert best["loss"] == 0.3

    def test_serialization(self):
        """Should serialize and deserialize correctly."""
        logger = MetricsLogger()

        logger.log(1, "train", {"loss": 0.5})
        logger.log(2, "train", {"loss": 0.4})

        # Serialize
        data = logger.to_dict()

        # Deserialize
        restored = MetricsLogger.from_dict(data)

        assert len(restored) == 2
        assert restored.latest()["epoch"] == 2


class TestMetricsLoggerEdgeCases:
    """Test edge cases."""

    def test_empty_history_best(self):
        """Best on empty history should return None."""
        logger = MetricsLogger()

        best = logger.get_best("iou")
        assert best is None

    def test_empty_history_latest(self):
        """Latest on empty history should return None."""
        logger = MetricsLogger()

        assert logger.latest() is None
        assert logger.latest("train") is None

    def test_repr(self):
        """repr should show entry count."""
        logger = MetricsLogger()
        logger.log(1, "train", {"loss": 0.5})

        assert "1" in repr(logger)

    def test_filter_nonexistent_split(self):
        """Filter for nonexistent split should return empty."""
        logger = MetricsLogger()
        logger.log(1, "train", {"loss": 0.5})

        history = logger.get_history("val")
        assert len(history) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])