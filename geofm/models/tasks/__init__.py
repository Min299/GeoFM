"""geofm.models.tasks

Task definitions and registry.
"""
from geofm.models.tasks.task_registry import (
    TASK_REGISTRY,
    get_task_config,
)

__all__ = [
    "TASK_REGISTRY",
    "get_task_config",
]