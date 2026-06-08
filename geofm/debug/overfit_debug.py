"""geofm.debug.overfit_debug

Debug utilities for diagnosing overfitting issues.
"""
from __future__ import annotations

from typing import List, Dict, Optional
import numpy as np


class OverfitDebug:
    """Debug tool for tracking overfitting.

    Tracks training and validation losses to detect overfitting.

    Usage:
        debug = OverfitDebug()
        debug.log_loss(0.5, mode="train")
        debug.log_loss(0.6, mode="val")
        summary = debug.summary()
    """

    def __init__(self, patience: int = 5):
        """Initialize overfit debug.

        Args:
            patience: Number of epochs to wait before declaring overfit
        """
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.patience = patience
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.overfit_detected = False

    def log_loss(
        self,
        loss: float,
        mode: str = "train",
    ) -> None:
        """Log a loss value.

        Args:
            loss: Loss value
            mode: "train" or "val"
        """
        if mode == "train":
            self.train_losses.append(float(loss))
        elif mode == "val":
            self.val_losses.append(float(loss))

            # Check for overfit
            if loss < self.best_val_loss:
                self.best_val_loss = loss
                self.best_epoch = len(self.val_losses) - 1
            else:
                # Check if we've been worse for patience epochs
                epochs_without_improvement = len(self.val_losses) - 1 - self.best_epoch
                if epochs_without_improvement >= self.patience:
                    self.overfit_detected = True

    def summary(self) -> Dict:
        """Get summary of overfitting status.

        Returns:
            Dictionary with overfitting analysis
        """
        if not self.train_losses or not self.val_losses:
            return {
                "status": "no_data",
                "train_losses": [],
                "val_losses": [],
            }

        train_reduction = self.train_losses[0] - self.train_losses[-1]
        val_reduction = self.val_losses[0] - self.val_losses[-1]

        # Detect if train loss decreased but val loss increased
        train_decreased = train_reduction > 0
        val_increased = val_reduction < 0

        overfitting = train_decreased and val_increased

        return {
            "status": "overfitting" if overfitting else "healthy",
            "overfit_detected": self.overfit_detected,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_reduction": train_reduction,
            "val_reduction": val_reduction,
            "gap": self.val_losses[-1] - self.train_losses[-1] if self.train_losses and self.val_losses else 0,
        }

    def plot_data(self) -> Dict[str, List]:
        """Get data for plotting.

        Returns:
            Dictionary with plot data
        """
        return {
            "train": {
                "x": list(range(len(self.train_losses))),
                "y": self.train_losses,
            },
            "val": {
                "x": list(range(len(self.val_losses))),
                "y": self.val_losses,
            },
        }


class GradientDebug:
    """Debug tool for tracking gradient health."""

    def __init__(self):
        """Initialize gradient debug."""
        self.gradient_norms: List[float] = []
        self.has_nan_gradients = False

    def log_gradients(self, model) -> None:
        """Log gradient norms from model.

        Args:
            model: PyTorch model
        """
        total_norm = 0.0

        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

        total_norm = total_norm ** 0.5
        self.gradient_norms.append(total_norm)

        # Check for NaN
        if np.isnan(total_norm):
            self.has_nan_gradients = True

    def summary(self) -> Dict:
        """Get gradient summary.

        Returns:
            Dictionary with gradient analysis
        """
        if not self.gradient_norms:
            return {
                "status": "no_data",
            }

        return {
            "status": "unstable" if self.has_nan_gradients else "healthy",
            "has_nan": self.has_nan_gradients,
            "gradient_norms": self.gradient_norms,
            "mean_norm": np.mean(self.gradient_norms),
            "max_norm": np.max(self.gradient_norms),
            "min_norm": np.min(self.gradient_norms),
        }


class ParameterDebug:
    """Debug tool for tracking parameter statistics."""

    def __init__(self):
        """Initialize parameter debug."""
        self.snapshots: Dict[str, List] = {}

    def snapshot(self, name: str, model) -> None:
        """Take a snapshot of model parameters.

        Args:
            name: Snapshot name
            model: PyTorch model
        """
        stats = {}

        for n, p in model.named_parameters():
            stats[n] = {
                "mean": p.data.mean().item(),
                "std": p.data.std().item(),
                "min": p.data.min().item(),
                "max": p.data.max().item(),
                "norm": p.data.norm(2).item(),
            }

        self.snapshots[name] = stats

    def compare(
        self,
        name1: str,
        name2: str,
    ) -> Dict:
        """Compare two snapshots.

        Args:
            name1: First snapshot name
            name2: Second snapshot name

        Returns:
            Dictionary with comparison results
        """
        if name1 not in self.snapshots or name2 not in self.snapshots:
            return {"status": "missing_snapshot"}

        s1 = self.snapshots[name1]
        s2 = self.snapshots[name2]

        changes = {}

        for param_name in s1:
            if param_name in s2:
                norm_change = s2[param_name]["norm"] - s1[param_name]["norm"]
                changes[param_name] = {
                    "norm_change": norm_change,
                    "relative_change": norm_change / (s1[param_name]["norm"] + 1e-8),
                }

        return {
            "status": "ok",
            "changes": changes,
        }