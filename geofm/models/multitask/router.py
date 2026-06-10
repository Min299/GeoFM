"""geofm.models.multitask.router

Routes batches to the correct task model.
"""
from __future__ import annotations


class TaskRouter:
    """
    Routes batches to the correct task model.
    """

    def __init__(
        self,
        model,
    ):
        """Initialize the router with a model.

        Args:
            model: The model to route requests to
        """
        self.model = model

    def forward(
        self,
        batch,
        task_name: str,
    ):
        """Forward pass through the model.

        Args:
            batch: Input batch
            task_name: Name of the task

        Returns:
            Model output
        """
        return self.model(
            batch,
            task_name=task_name,
        )

    __call__ = forward
