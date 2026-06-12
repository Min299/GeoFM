"""geofm.models.tasks.task_factory

Factory for building task-specific models.
"""
from __future__ import annotations

from geofm.models.tasks.segmentation_model import (
    SegmentationModel,
)

from geofm.models.tasks.classification_model import (
    ClassificationModel,
)


class TaskFactory:
    """Factory for building task-specific models."""

    @staticmethod
    def build(
        task_name,
        backbone,
        head,
    ):
        """Build task model.

        Args:
            task_name: Task name ("segmentation" or "classification")
            backbone: Backbone model
            head: Task head (decoder or classifier)

        Returns:
            Task-specific model

        Raises:
            ValueError: If task_name is unknown
        """
        if task_name == "segmentation":

            return SegmentationModel(
                backbone,
                head,
            )

        if task_name == "classification":

            return ClassificationModel(
                backbone,
                head,
            )

        raise ValueError(
            f"Unknown task: {task_name}"
        )