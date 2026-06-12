"""geofm.training.epoch_runner

Epoch-level training and validation runner.
"""
from __future__ import annotations

from geofm.training.train_step import (
    train_step,
)

from geofm.training.eval_step import (
    eval_step,
)


class EpochRunner:
    """Epoch-level runner for training and validation loops.

    Provides train_epoch and validate_epoch methods that iterate
    over a dataloader and execute the appropriate step functions.
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
    ):
        """Initialize the EpochRunner.

        Args:
            model: The PyTorch model to train/evaluate
            optimizer: The optimizer for training steps
            criterion: The loss function
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def train_epoch(
        self,
        loader,
        task: str = "flood",
    ):
        """Run one training epoch.

        Args:
            loader: DataLoader providing batches
            task: Task name for the model forward pass

        Returns:
            Average loss across all batches
        """
        self.model.train()

        losses = []

        for batch in loader:
            loss = train_step(
                self.model,
                batch,
                self.optimizer,
                self.criterion,
                task=task,
            )
            losses.append(loss)

        return sum(losses) / len(losses) if losses else 0.0

    def validate_epoch(
        self,
        loader,
        task: str = "flood",
    ):
        """Run one validation epoch.

        Args:
            loader: DataLoader providing batches
            task: Task name for the model forward pass

        Returns:
            Average loss across all batches
        """
        self.model.eval()

        losses = []

        for batch in loader:
            loss = eval_step(
                self.model,
                batch,
                self.criterion,
                task=task,
            )
            losses.append(loss)

        return sum(losses) / len(losses) if losses else 0.0