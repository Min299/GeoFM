"""geofm.evaluation

Evaluation metrics for GeoFM tasks.
"""
from geofm.evaluation.segmentation_metrics import SegmentationMetrics
from geofm.evaluation.classification_metrics import ClassificationMetrics
from geofm.evaluation.regression_metrics import RegressionMetrics
from geofm.evaluation.metric_registry import (
    get_metrics,
    get_primary_metric,
    is_segmentation_task,
    is_classification_task,
    is_regression_task,
)

__all__ = [
    # Metrics
    "SegmentationMetrics",
    "ClassificationMetrics",
    "RegressionMetrics",
    # Registry
    "get_metrics",
    "get_primary_metric",
    "is_segmentation_task",
    "is_classification_task",
    "is_regression_task",
]