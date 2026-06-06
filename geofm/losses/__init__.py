"""geofm.losses

Loss functions for GeoFM.
"""
from geofm.losses.segmentation import (
    DiceLoss,
    CombinedLoss,
    FocalLoss,
    BoundaryLoss,
)
from geofm.losses.classification import (
    LabelSmoothingCrossEntropy,
    WeightedCrossEntropy,
    BCEWithLogitsLoss,
)
from geofm.losses.regression import (
    MSELoss,
    MAELoss,
    HuberLoss,
    SmoothL1Loss,
    QuantileLoss,
)
from geofm.losses.multitask import (
    MultiTaskLoss,
    DynamicWeightAverage,
    GradientNormBalancing,
)

__all__ = [
    # Segmentation
    "DiceLoss",
    "CombinedLoss",
    "FocalLoss",
    "BoundaryLoss",
    # Classification
    "LabelSmoothingCrossEntropy",
    "WeightedCrossEntropy",
    "BCEWithLogitsLoss",
    # Regression
    "MSELoss",
    "MAELoss",
    "HuberLoss",
    "SmoothL1Loss",
    "QuantileLoss",
    # Multi-task
    "MultiTaskLoss",
    "DynamicWeightAverage",
    "GradientNormBalancing",
]
