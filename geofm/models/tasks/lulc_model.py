"""geofm.models.tasks.lulc_model

Land Use / Land Cover classification model.
"""
from __future__ import annotations

from geofm.models.tasks.segmentation_model import (
    SegmentationModel,
)


class LULCModel(
    SegmentationModel
):
    """Land Use / Land Cover classification model.

    Specialized segmentation model for LULC classification tasks.
    Inherits from SegmentationModel - task-specific customization
    can be added as needed.
    """
    pass