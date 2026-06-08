"""geofm.training.trainer

Trainer class for model training.
"""
from __future__ import annotations

from typing import Optional, List, Callable

import torch
import torch.nn as nn


class Trainer:
    """Trainer for model training.

    Usage:
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
        )

        for epoch in range(epochs):
            train_loss = trainer.train_epoch(train_loader)
            val_loss = trainer.eval_epoch(val_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: Optional[str] = None,
        grad_clip: Optional[float] = None,
    ):
        """Initialize trainer.

        Args:
            model: The model to train
            optimizer: Optimizer
            criterion: Loss function
            device: Device to use (cpu, cuda)
            grad_clip: Maximum gradient norm for clipping
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.grad_clip = grad_clip

        self.model.to(self.device)

    def train_step(self, batch: dict, task: str = "flood") -> float:
        """Execute one training step.

        Args:
            batch: Batch dictionary with 'inputs' and 'mask'
            task: Task name

        Returns:
            Loss value
        """
        self.model.train()

        inputs = batch["inputs"].to(self.device)
        targets = batch["mask"].to(self.device)

        self.optimizer.zero_grad()

        logits = self.model(inputs, task_name=task)
        loss = self.criterion(logits, targets)

        loss.backward()

        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.optimizer.step()

        return loss.item()

    def eval_step(self, batch: dict, task: str = "flood") -> float:
        """Execute one evaluation step.

        Args:
            batch: Batch dictionary
            task: Task name

        Returns:
            Loss value
        """
        self.model.eval()

        with torch.no_grad():
            inputs = batch["inputs"].to(self.device)
            targets = batch["mask"].to(self.device)

            logits = self.model(inputs, task_name=task)
            loss = self.criterion(logits, targets)

        return loss.item()

    def train_epoch(self, loader, task: str = "flood") -> float:
        """Train for one epoch.

        Args:
            loader: Data loader
            task: Task name

        Returns:
            Mean training loss
        """
        losses = []

        for batch in loader:
            loss = self.train_step(batch, task)
            losses.append(loss)

        return sum(losses) / len(losses) if losses else 0.0

    def eval_epoch(self, loader, task: str = "flood") -> float:
        """Evaluate for one epoch.

        Args:
            loader: Data loader
            task: Task name

        Returns:
            Mean evaluation loss
        """
        losses = []

        self.model.eval()

        with torch.no_grad():
            for batch in loader:
                loss = self.eval_step(batch, task)
                losses.append(loss)

        return sum(losses) / len(losses) if losses else 0.0

    def save_checkpoint(self, path: str, epoch: int, **kwargs):
        """Save training checkpoint.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            **kwargs: Additional items to save
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            **kwargs,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load training checkpoint.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["epoch"]


class SimpleTrainer:
    """Simplified trainer without device management."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def step(self, batch: dict, task: str = "flood") -> float:
        """Single training step.

        Args:
            batch: Batch dictionary
            task: Task name

        Returns:
            Loss value
        """
        self.model.train()

        self.optimizer.zero_grad()

        logits = self.model(batch["inputs"], task_name=task)
        loss = self.criterion(logits, batch["mask"])

        loss.backward()
        self.optimizer.step()

        return loss.item()