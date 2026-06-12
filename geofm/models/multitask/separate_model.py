"""geofm.models.multitask.separate_model

Independent model per task.

flood -> model
burn  -> model
lulc  -> model
crop  -> model
ndvi  -> model
"""
from __future__ import annotations

import torch.nn as nn


class SeparateGeoFM(nn.Module):
    """
    Independent model per task.
    """

    def __init__(
        self,
        task_models: dict[str, nn.Module],
    ):
        super().__init__()

        self.task_models = nn.ModuleDict(
            task_models
        )

    def available_tasks(self):
        """Return list of available task names."""
        return list(
            self.task_models.keys()
        )

    def forward(
        self,
        batch,
        task_name: str,
    ):
        """Forward pass through the appropriate task model.

        Args:
            batch: Input batch
            task_name: Name of the task

        Returns:
            Model output for the specified task

        Raises:
            KeyError: If task_name is not available
        """
        if task_name not in self.task_models:

            raise KeyError(
                f"Unknown task: {task_name}"
            )

        return self.task_models[
            task_name
        ](
            batch
        )
