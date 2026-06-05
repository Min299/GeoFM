"""geofm.trainers.base_trainer

Abstract base trainer for all training scenarios.
Research projects become messy without this.

All trainers should inherit from BaseTrainer and implement:
- train()
- validate()
- test()
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

import torch
from torch.utils.data import DataLoader


class BaseTrainer(ABC):
    """Abstract base trainer.

    Defines the interface that all trainers must implement.
    Ensures consistent training loop across experiments.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        device: str = "auto"
    ):
        """Initialize base trainer."""
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model.to(self.device)
        self.current_epoch = 0
        self.best_val_loss = float("inf")

    @abstractmethod
    def train(self):
        """Run one epoch of training."""
        raise NotImplementedError

    @abstractmethod
    def validate(self):
        """Run validation. Returns: Validation metrics dict"""
        raise NotImplementedError

    @abstractmethod
    def test(self):
        """Run testing. Returns: Test metrics dict"""
        raise NotImplementedError

    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint."""
        torch.save({
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "best_val_loss": self.best_val_loss,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]

    def get_metrics(self) -> Dict[str, Any]:
        """Get current training metrics."""
        return {
            "epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss,
            "device": self.device,
        }
