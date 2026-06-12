"""geofm.checkpoint.checkpoint_manager

Checkpoint management for training.

Provides utilities for saving and managing checkpoints during training.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Union, Optional, List
import torch
import torch.nn as nn

from geofm.checkpoint.model_saver import ModelSaver
from geofm.checkpoint.model_loader import ModelLoader


class CheckpointManager:
    """Manager for saving and loading checkpoints during training.

    Handles checkpoint saving, loading, and cleanup.

    Usage:
        manager = CheckpointManager("checkpoints/my_exp")

        # Save checkpoint
        manager.save_epoch(model, optimizer, epoch=5, metrics={"loss": 0.5})

        # Load checkpoint
        manager.load_latest(model, optimizer)
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        experiment_name: Optional[str] = None,
        max_checkpoints: int = 5,
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoints
            experiment_name: Optional experiment name for naming
            max_checkpoints: Maximum checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.experiment_name = experiment_name
        self.max_checkpoints = max_checkpoints
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _get_checkpoint_path(self, name: str) -> Path:
        """Get full path for a checkpoint.

        Args:
            name: Checkpoint name

        Returns:
            Full path
        """
        return self.checkpoint_dir / f"{name}.pt"

    def save_epoch(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
        name: Optional[str] = None,
    ) -> Path:
        """Save checkpoint for an epoch.

        Args:
            model: PyTorch model
            optimizer: Optional optimizer
            epoch: Epoch number
            metrics: Optional metrics dict
            name: Optional custom name

        Returns:
            Path to saved checkpoint
        """
        if name is None:
            if epoch is not None:
                name = f"epoch_{epoch}"
            else:
                name = "checkpoint"

        path = self._get_checkpoint_path(name)

        ModelSaver.save_checkpoint(
            model=model,
            optimizer=optimizer,
            path=path,
            epoch=epoch,
            metrics=metrics,
        )

        self._cleanup_old_checkpoints()

        return path

    def save_best(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Path:
        """Save best model checkpoint.

        Args:
            model: PyTorch model
            optimizer: Optional optimizer
            metrics: Optional metrics dict

        Returns:
            Path to saved checkpoint
        """
        path = self._get_checkpoint_path("best")

        ModelSaver.save_checkpoint(
            model=model,
            optimizer=optimizer,
            path=path,
            metrics=metrics,
            extra={"is_best": True},
        )

        return path

    def save_latest(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Path:
        """Save latest checkpoint.

        Args:
            model: PyTorch model
            optimizer: Optional optimizer
            epoch: Optional epoch number
            metrics: Optional metrics dict

        Returns:
            Path to saved checkpoint
        """
        path = self._get_checkpoint_path("latest")

        ModelSaver.save_checkpoint(
            model=model,
            optimizer=optimizer,
            path=path,
            epoch=epoch,
            metrics=metrics,
        )

        return path

    def load_latest(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        map_location: str = "cpu",
    ) -> Optional[Dict[str, Any]]:
        """Load the latest checkpoint.

        Args:
            model: PyTorch model
            optimizer: Optional optimizer to load state
            map_location: Device to load to

        Returns:
            Checkpoint dict or None if no checkpoint
        """
        latest_path = self._get_checkpoint_path("latest")

        if not latest_path.exists():
            return None

        return self.load_checkpoint(
            model=model,
            optimizer=optimizer,
            path=latest_path,
            map_location=map_location,
        )

    def load_best(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        map_location: str = "cpu",
    ) -> Optional[Dict[str, Any]]:
        """Load the best checkpoint.

        Args:
            model: PyTorch model
            optimizer: Optional optimizer to load state
            map_location: Device to load to

        Returns:
            Checkpoint dict or None if no checkpoint
        """
        best_path = self._get_checkpoint_path("best")

        if not best_path.exists():
            return None

        return self.load_checkpoint(
            model=model,
            optimizer=optimizer,
            path=best_path,
            map_location=map_location,
        )

    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        path: Optional[Path] = None,
        map_location: str = "cpu",
    ) -> Dict[str, Any]:
        """Load a specific checkpoint.

        Args:
            model: PyTorch model
            optimizer: Optional optimizer to load state
            path: Path to checkpoint (if None, loads latest)
            map_location: Device to load to

        Returns:
            Checkpoint dict
        """
        if path is None:
            path = self._get_checkpoint_path("latest")

        checkpoint = ModelLoader.load_checkpoint(path, map_location)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint

    def list_checkpoints(self) -> List[Path]:
        """List all checkpoints in directory.

        Returns:
            List of checkpoint paths
        """
        return sorted(self.checkpoint_dir.glob("*.pt"))

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints if over limit."""
        checkpoints = self.list_checkpoints()

        # Keep latest and best
        protected = {"latest.pt", "best.pt"}

        # Sort by modification time (oldest first)
        checkpoints.sort(key=lambda p: p.stat().st_mtime)

        # Remove oldest if over limit
        while len(checkpoints) > self.max_checkpoints:
            oldest = checkpoints.pop(0)
            if oldest.name not in protected:
                oldest.unlink()

    def get_checkpoint_info(
        self,
        path: Path,
    ) -> Dict[str, Any]:
        """Get info about a checkpoint.

        Args:
            path: Path to checkpoint

        Returns:
            Info dict
        """
        return ModelLoader.get_checkpoint_info(path)

    def delete_checkpoint(self, name: str) -> bool:
        """Delete a checkpoint by name.

        Args:
            name: Checkpoint name (without .pt extension)

        Returns:
            True if deleted
        """
        path = self._get_checkpoint_path(name)

        if path.exists():
            path.unlink()
            return True

        return False

    def clear_all(self) -> None:
        """Delete all checkpoints."""
        for path in self.list_checkpoints():
            path.unlink()