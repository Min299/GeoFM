"""geofm.models.heads

Task-specific output heads.
"""
from geofm.models.heads.segmentation_head import SegmentationHead
from geofm.models.heads.classification_head import ClassificationHead
from geofm.models.heads.regression_head import RegressionHead

__all__ = [
    "SegmentationHead",
    "ClassificationHead",
    "RegressionHead",
]