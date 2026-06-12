"""geofm.builders.independent_model_builder

Builder for creating independent per-task models.
"""
from __future__ import annotations

from geofm.models.multitask.separate_model import (
    SeparateGeoFM,
)


class IndependentModelBuilder:
    """Builder for creating SeparateGeoFM with independent task models."""

    def __init__(self):
        """Initialize the builder."""
        self.task_models = {}

    def add_task_model(
        self,
        task_name,
        model,
    ):
        """Add a task-specific model.

        Args:
            task_name: Name of the task (e.g., "flood", "burn")
            model: The model for this task

        Returns:
            self for chaining
        """
        self.task_models[
            task_name
        ] = model

        return self

    def build(self):
        """Build the SeparateGeoFM model.

        Returns:
            SeparateGeoFM instance with all registered task models

        Raises:
            ValueError: If no task models have been registered
        """
        if len(
            self.task_models
        ) == 0:

            raise ValueError(
                "No task models registered."
            )

        return SeparateGeoFM(
            self.task_models
        )