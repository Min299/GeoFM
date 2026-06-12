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
from geofm.models.tasks.task_factory import (
    TaskFactory,
)
from geofm.models.tasks.flood_model import (
    FloodModel,
)
from geofm.models.tasks.burnscar_model import (
    BurnScarModel,
)
from geofm.models.tasks.lulc_model import (
    LULCModel,
)

__all__ = [
    "TASK_REGISTRY",
    "get_task_config",
    "SegmentationModel",
    "ClassificationModel",
    "TaskFactory",
    "FloodModel",
    "BurnScarModel",
    "LULCModel",
]