"""geofm.trainers.finetune_trainer

Fine-tuning trainer for Experiment 1:
- Flood + TerraMind Base + Full FT
- Flood + TerraMind Base + LoRA

This trainer drives Phase 1 experiments comparing
full fine-tuning vs LoRA.
"""
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from geofm.trainers.base_trainer import BaseTrainer


class FineTuneTrainer(BaseTrainer):
    """Trainer for fine-tuning experiments.

    Supports both Full FT and LoRA modes.
    Designed for the comparison experiments in Phase 1.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "auto",
        max_epochs: int = 100,
        early_stopping_patience: int = 10
    ):
        """Initialize fine-tune trainer."""
        super().__init__(model, train_loader, val_loader, test_loader, device)

        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=0.1
        )

        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.patience_counter = 0
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    def train(self) -> Dict[str, Any]:
        """Run training for max_epochs or until early stopping."""
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.max_epochs):
            self.current_epoch = epoch

            train_loss = self._train_epoch()
            history["train_loss"].append(train_loss)

            val_loss = self.validate()["loss"]
            history["val_loss"].append(val_loss)

            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        return history

    def _train_epoch(self) -> float:
        """Train one epoch."""
        if self.train_loader is None:
            return 0.0

        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            if isinstance(batch, dict):
                modalities = batch.get("modalities", {})
                labels = batch.get("label")
            else:
                modalities = {}
                labels = batch

            self.optimizer.zero_grad()
            outputs = self.model(modalities)

            if labels is not None:
                loss = self.criterion(outputs, labels.to(self.device))
            else:
                loss = torch.tensor(0.0, device=self.device)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def validate(self) -> Dict[str, Any]:
        """Run validation."""
        if self.val_loader is None:
            return {"loss": 0.0}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                if isinstance(batch, dict):
                    modalities = batch.get("modalities", {})
                    labels = batch.get("label")
                else:
                    modalities = {}
                    labels = batch

                outputs = self.model(modalities)

                if labels is not None:
                    loss = self.criterion(outputs, labels.to(self.device))
                    total_loss += loss.item()
                    num_batches += 1

        return {"loss": total_loss / max(num_batches, 1)}

    def test(self) -> Dict[str, Any]:
        """Run testing."""
        if self.test_loader is None:
            return {"loss": 0.0, "metrics": {}}

        self.model.eval()
        return self.validate()
