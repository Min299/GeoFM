"""geofm.models.tasks.burnscar_model

Burn scar segmentation model.
"""
from __future__ import annotations

from geofm.models.tasks.segmentation_model import (
    SegmentationModel,
)


class BurnScarModel(
    SegmentationModel
):
    """Burn scar segmentation model.

    Specialized segmentation model for burn scar detection tasks.
    Inherits from SegmentationModel - task-specific customization
    can be added as needed.
    """
    pass