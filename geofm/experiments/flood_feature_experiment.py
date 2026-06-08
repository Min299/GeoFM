"""geofm.experiments.flood_feature_experiment

Flood feature adapter experiment.
"""
from __future__ import annotations

from typing import Optional, Dict

import torch


class FloodFeatureExperiment:
    """Experiment runner for flood segmentation with feature adapter.

    This is the simplest experiment:
    - Single task: Flood
    - Single adapter: Feature Adapter
    - No LoRA, no Hybrid, no Full FT
    """

    def __init__(
        self,
        model: torch.nn.Module,
        trainer,
        device: Optional[str] = None,
    ):
        """Initialize experiment.

        Args:
            model: The model to train
            trainer: Trainer instance
            device: Device to use
        """
        self.model = model
        self.trainer = trainer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.current_epoch = 0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": [],
        }

    def run(
        self,
        train_loader,
        val_loader=None,
        num_epochs: int = 10,
        task: str = "flood",
    ) -> Dict:
        """Run the experiment.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            num_epochs: Number of epochs to train
            task: Task name

        Returns:
            Training history
        """
        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_loss = self.trainer.train_epoch(train_loader, task)
            self.history["train_loss"].append(train_loss)

            # Validate
            if val_loader is not None:
                val_loss = self.trainer.eval_epoch(val_loader, task)
                self.history["val_loss"].append(val_loss)

            # Log
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}", end="")
            if val_loader is not None:
                print(f", val_loss={val_loss:.4f}")
            else:
                print()

        return self.history

    def run_single_epoch(self, train_loader, task: str = "flood") -> float:
        """Run a single training epoch.

        Args:
            train_loader: Training data loader
            task: Task name

        Returns:
            Mean training loss
        """
        return self.trainer.train_epoch(train_loader, task)

    def get_history(self) -> Dict:
        """Get training history.

        Returns:
            Training history dictionary
        """
        return self.history

    def reset_history(self):
        """Reset training history."""
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": [],
        }
        self.current_epoch = 0


class ExperimentConfig:
    """Configuration for experiments."""

    def __init__(
        self,
        task: str = "flood",
        adapter: str = "feature",
        epochs: int = 10,
        batch_size: int = 8,
        lr: float = 1e-4,
        optimizer: str = "adamw",
        **kwargs,
    ):
        self.task = task
        self.adapter = adapter
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = optimizer
        self.extra = kwargs

    @classmethod
    def from_dict(cls, config: dict) -> "ExperimentConfig":
        """Create config from dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            ExperimentConfig instance
        """
        return cls(**config)

    def to_dict(self) -> dict:
        """Convert config to dictionary.

        Returns:
            Configuration dictionary
        """
        result = {
            "task": self.task,
            "adapter": self.adapter,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "optimizer": self.optimizer,
        }
        result.update(self.extra)
        return result


def create_flood_feature_experiment(
    model,
    lr: float = 1e-4,
    epochs: int = 10,
) -> FloodFeatureExperiment:
    """Create a flood feature experiment.

    Args:
        model: The model to train
        lr: Learning rate
        epochs: Number of epochs

    Returns:
        FloodFeatureExperiment instance
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    from geofm.training.trainer import SimpleTrainer
    trainer = SimpleTrainer(model, optimizer, criterion)

    return FloodFeatureExperiment(model, trainer)