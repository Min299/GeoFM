"""geofm.checkpoint.model_saver

Model saving utilities.

Handles saving PyTorch models and checkpoints.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Union, Optional
import torch
import json


class ModelSaver:
    """Utility for saving PyTorch models.

    Usage:
        ModelSaver.save(model, "checkpoints/model.pt")
        ModelSaver.save_checkpoint(model, optimizer, "checkpoints/ckpt.pt")
    """

    @staticmethod
    def save(
        model: torch.nn.Module,
        path: Union[str, Path],
    ) -> None:
        """Save model state dict.

        Args:
            model: PyTorch model
            path: Output path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), path)

    @staticmethod
    def save_checkpoint(
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        path: Union[str, Path],
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save full checkpoint with metadata.

        Args:
            model: PyTorch model
            optimizer: Optional optimizer
            path: Output path
            epoch: Optional epoch number
            metrics: Optional metrics dict
            extra: Optional extra data
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": model.state_dict(),
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if epoch is not None:
            checkpoint["epoch"] = epoch

        if metrics is not None:
            checkpoint["metrics"] = metrics

        if extra is not None:
            checkpoint["extra"] = extra

        torch.save(checkpoint, path)

    @staticmethod
    def save_model_only(
        model: torch.nn.Module,
        path: Union[str, Path],
        save_architecture: bool = True,
    ) -> None:
        """Save model without full checkpoint.

        Args:
            model: PyTorch model
            path: Output path
            save_architecture: Whether to save architecture info
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "state_dict": model.state_dict(),
        }

        if save_architecture:
            data["architecture"] = {
                "class": model.__class__.__name__,
                "module": model.__class__.__module__,
            }

        torch.save(data, path)


def save_model(
    model: torch.nn.Module,
    path: Union[str, Path],
) -> None:
    """Convenience function to save model.

    Args:
        model: PyTorch model
        path: Output path
    """
    ModelSaver.save(model, path)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: Union[str, Path],
    **kwargs,
) -> None:
    """Convenience function to save checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        path: Output path
        **kwargs: Additional checkpoint data
    """
    ModelSaver.save_checkpoint(model, optimizer, path, **kwargs)