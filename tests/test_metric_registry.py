"""tests/test_metric_registry.py

Tests for metric registry.
"""
import pytest

from geofm.evaluation.metric_registry import (
    get_metrics,
    get_primary_metric,
    is_segmentation_task,
    is_classification_task,
    is_regression_task,
    METRIC_REGISTRY,
)


class TestGetMetrics:
    """Test get_metrics function."""

    def test_flood_metrics(self):
        """Flood should have segmentation metrics."""
        metrics = get_metrics("flood")

        assert "iou" in metrics
        assert "dice" in metrics
        assert "f1" in metrics

    def test_burn_metrics(self):
        """Burn should have segmentation metrics."""
        metrics = get_metrics("burn")

        assert "iou" in metrics

    def test_lulc_metrics(self):
        """LULC should have classification metrics."""
        metrics = get_metrics("lulc")

        assert "accuracy" in metrics

    def test_ndvi_metrics(self):
        """NDVI should have regression metrics."""
        metrics = get_metrics("ndvi")

        assert "rmse" in metrics
        assert "mae" in metrics

    def test_unknown_task(self):
        """Unknown task should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown task"):
            get_metrics("unknown")


class TestGetPrimaryMetric:
    """Test get_primary_metric function."""

    def test_segmentation_primary(self):
        """Flood and burn should use IoU."""
        assert get_primary_metric("flood") == "iou"
        assert get_primary_metric("burn") == "iou"

    def test_classification_primary(self):
        """LULC should use accuracy."""
        assert get_primary_metric("lulc") == "accuracy"

    def test_regression_primary(self):
        """NDVI should use RMSE."""
        assert get_primary_metric("ndvi") == "rmse"


class TestTaskTypeDetection:
    """Test task type detection functions."""

    def test_segmentation_tasks(self):
        """Should identify segmentation tasks."""
        assert is_segmentation_task("flood") is True
        assert is_segmentation_task("burn") is True
        assert is_segmentation_task("lulc") is False

    def test_classification_tasks(self):
        """Should identify classification tasks."""
        assert is_classification_task("lulc") is True
        assert is_classification_task("crop") is True
        assert is_classification_task("flood") is False

    def test_regression_tasks(self):
        """Should identify regression tasks."""
        assert is_regression_task("ndvi") is True
        assert is_regression_task("flood") is False


class TestMetricRegistry:
    """Test METRIC_REGISTRY constant."""

    def test_all_tasks_defined(self):
        """All known tasks should be in registry."""
        expected_tasks = ["flood", "burn", "lulc", "crop", "ndvi"]

        for task in expected_tasks:
            assert task in METRIC_REGISTRY

    def test_metrics_not_empty(self):
        """All tasks should have at least one metric."""
        for task, metrics in METRIC_REGISTRY.items():
            assert len(metrics) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])