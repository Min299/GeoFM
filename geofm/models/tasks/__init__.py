"""geofm.models.tasks

Task definitions, models, and registry.
"""
from geofm.models.tasks.task_registry import (
    TASK_REGISTRY,
    get_task_config,
)
from geofm.models.tasks.segmentation_model import (
    SegmentationModel,
)
from geofm.models.tasks.classification_model import (
    ClassificationModel,
)

__all__ = [
    "TASK_REGISTRY",
    "get_task_config",
    "SegmentationModel",
    "ClassificationModel",
]