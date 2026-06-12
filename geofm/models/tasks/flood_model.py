"""geofm.models.tasks.flood_model

Flood segmentation model.
"""
from __future__ import annotations

from geofm.models.tasks.segmentation_model import (
    SegmentationModel,
)


class FloodModel(
    SegmentationModel
):
    """Flood segmentation model.

    Specialized segmentation model for flood detection tasks.
    Inherits from SegmentationModel - task-specific customization
    can be added as needed.
    """
    pass