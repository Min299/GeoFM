"""geofm.trainers.shared_trainer

Shared training loop implementation.
"""
import torch
import torch.nn as nn
from typing import Optional, Callable


class SharedTrainer:
    """Shared trainer for GeoFM experiments.

    This is a simple trainer that can be replaced with
    PyTorch Lightning or Accelerate later.

    Usage:
        trainer = SharedTrainer(
            optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
            train_loader=train_loader,
            val_loader=val_loader,
        )
        trainer.fit(model, criterion, epochs=50)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        gradient_clip: Optional[float] = None,
        accumulation_steps: int = 1,
    ):
        """Initialize trainer.

        Args:
            optimizer: Optimizer instance
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            device: Device to use
            gradient_clip: Gradient clipping value (optional)
            accumulation_steps: Gradient accumulation steps
        """
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.gradient_clip = gradient_clip
        self.accumulation_steps = accumulation_steps

    def to_device(self, batch):
        """Move batch to device.

        Args:
            batch: Data batch

        Returns:
            Batch on device
        """
        if isinstance(batch, dict):
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return [x.to(self.device) if isinstance(x, torch.Tensor) else x
                    for x in batch]
        elif isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        return batch

    def train_step(
        self,
        model: nn.Module,
        batch: dict,
        criterion: Optional[Callable] = None,
    ) -> dict:
        """Single training step.

        Args:
            model: Model
            batch: Batch data
            criterion: Loss function

        Returns:
            Metrics dictionary
        """
        batch = self.to_device(batch)

        outputs = model(batch)

        # Compute loss
        if criterion is not None:
            target = batch.get("target", batch.get("labels"))
            if target is not None:
                loss = criterion(outputs, target)
            else:
                loss = outputs.mean()  # Dummy loss for testing
        else:
            loss = outputs.mean()

        # Scale loss for gradient accumulation
        loss = loss / self.accumulation_steps

        loss.backward()

        # Gradient clipping
        if self.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                self.gradient_clip,
            )

        return {"loss": loss.item() * self.accumulation_steps}

    def train_epoch(
        self,
        model: nn.Module,
        criterion: Optional[Callable] = None,
    ) -> dict:
        """Train for one epoch.

        Args:
            model: Model
            criterion: Loss function

        Returns:
            Metrics dictionary
        """
        model.train()

        total_loss = 0.0
        num_batches = len(self.train_loader)

        self.optimizer.zero_grad()

        for i, batch in enumerate(self.train_loader):
            # Forward and backward
            step_metrics = self.train_step(model, batch, criterion)

            total_loss += step_metrics.get("loss", 0.0)

            # Optimizer step
            if (i + 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

        # Handle remaining batches
        if (i + 1) % self.accumulation_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        avg_loss = total_loss / num_batches

        return {"loss": avg_loss}

    def validate(
        self,
        model: nn.Module,
        criterion: Optional[Callable] = None,
    ) -> dict:
        """Validate the model.

        Args:
            model: Model
            criterion: Loss function

        Returns:
            Metrics dictionary
        """
        if self.val_loader is None:
            return {}

        model.eval()

        total_loss = 0.0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for batch in self.val_loader:
                batch = self.to_device(batch)

                outputs = model(batch)

                if criterion is not None:
                    target = batch.get("target", batch.get("labels"))
                    if target is not None:
                        loss = criterion(outputs, target)
                    else:
                        loss = outputs.mean()
                else:
                    loss = outputs.mean()

                total_loss += loss.item()

        avg_loss = total_loss / num_batches

        return {"val_loss": avg_loss}

    def fit(
        self,
        model: nn.Module,
        criterion: Optional[Callable] = None,
        epochs: int = 1,
        callbacks: Optional[list] = None,
    ) -> dict:
        """Fit the model.

        Args:
            model: Model
            criterion: Loss function
            epochs: Number of epochs
            callbacks: Training callbacks

        Returns:
            Training history
        """
        history = []

        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(model, criterion)

            # Validate
            val_metrics = self.validate(model, criterion)

            # Combine
            metrics = {**train_metrics, **val_metrics}
            metrics["epoch"] = epoch

            history.append(metrics)

            # Callbacks
            if callbacks:
                for callback in callbacks:
                    if hasattr(callback, "on_epoch_end"):
                        callback.on_epoch_end(metrics)

            # Print
            loss_str = f"Epoch {epoch}: loss={train_metrics.get('loss', 0):.4f}"
            if val_metrics:
                loss_str += f", val_loss={val_metrics.get('val_loss', 0):.4f}"
            print(loss_str)

        return history


class TrainerCallback:
    """Base class for trainer callbacks."""

    def on_epoch_end(self, metrics: dict):
        """Called at the end of each epoch.

        Args:
            metrics: Epoch metrics
        """
        pass


class EarlyStopping(TrainerCallback):
    """Early stopping callback."""

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        monitor: str = "val_loss",
    ):
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait
            min_delta: Minimum improvement threshold
            monitor: Metric to monitor
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best = float("inf")
        self.counter = 0

    def on_epoch_end(self, metrics: dict):
        """Check for early stopping.

        Args:
            metrics: Epoch metrics
        """
        current = metrics.get(self.monitor, float("inf"))

        if current < self.best - self.min_delta:
            self.best = current
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print(f"Early stopping triggered after {self.counter} epochs without improvement")
            return True  # Should stop

        return False