"""geofm.evaluation

Evaluation metrics for GeoFM tasks.
"""
from geofm.evaluation.segmentation_metrics import SegmentationMetrics
from geofm.evaluation.classification_metrics import ClassificationMetrics
from geofm.evaluation.regression_metrics import RegressionMetrics

__all__ = [
    "SegmentationMetrics",
    "ClassificationMetrics",
    "RegressionMetrics",
]