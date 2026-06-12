"""geofm.checkpoint.resume

Resume training utilities.

Provides simple interface for resuming training from checkpoint.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Union, Optional, Tuple
import torch
import torch.nn as nn


def resume_checkpoint(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """Resume training from checkpoint.

    Loads model and optimizer state from checkpoint.

    Args:
        model: PyTorch model
        checkpoint_path: Path to checkpoint
        optimizer: Optional optimizer to load state
        map_location: Device to load to

    Returns:
        Checkpoint dictionary with metadata

    Usage:
        checkpoint = resume_checkpoint(
            model=model,
            checkpoint_path="checkpoints/latest.pt",
            optimizer=optimizer,
        )

        start_epoch = checkpoint.get("epoch", 0) + 1
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    # Load model state
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def resume_from_latest(
    model: nn.Module,
    checkpoint_dir: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str = "cpu",
) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
    """Resume from latest checkpoint in directory.

    Args:
        model: PyTorch model
        checkpoint_dir: Directory containing checkpoints
        optimizer: Optional optimizer to load state
        map_location: Device to load to

    Returns:
        Tuple of (checkpoint dict, start epoch) or (None, None)
    """
    checkpoint_dir = Path(checkpoint_dir)
    latest = checkpoint_dir / "latest.pt"

    if not latest.exists():
        return None, None

    checkpoint = resume_checkpoint(
        model=model,
        checkpoint_path=latest,
        optimizer=optimizer,
        map_location=map_location,
    )

    start_epoch = checkpoint.get("epoch", 0) + 1

    return checkpoint, start_epoch


def get_resume_info(
    checkpoint_path: Union[str, Path],
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """Get information needed to resume training.

    Args:
        checkpoint_path: Path to checkpoint
        map_location: Device to load to

    Returns:
        Dictionary with resume information
    """
    checkpoint_path = Path(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    return {
        "epoch": checkpoint.get("epoch"),
        "metrics": checkpoint.get("metrics", {}),
        "path": str(checkpoint_path),
    }


def auto_resume(
    model: nn.Module,
    checkpoint_dir: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str = "cpu",
) -> Tuple[bool, Optional[int]]:
    """Automatically resume from latest checkpoint if available.

    Args:
        model: PyTorch model
        checkpoint_dir: Directory containing checkpoints
        optimizer: Optional optimizer to load state
        map_location: Device to load to

    Returns:
        Tuple of (resumed, start_epoch)
    """
    checkpoint_dir = Path(checkpoint_dir)

    # Check for latest checkpoint
    latest = checkpoint_dir / "latest.pt"

    if not latest.exists():
        return False, None

    checkpoint = resume_checkpoint(
        model=model,
        checkpoint_path=latest,
        optimizer=optimizer,
        map_location=map_location,
    )

    start_epoch = checkpoint.get("epoch", 0) + 1

    return True, start_epoch


class ResumeManager:
    """Manager for handling training resume.

    Usage:
        manager = ResumeManager("checkpoints/my_exp", start_epoch=0)

        # At end of each epoch
        manager.save_checkpoint(model, optimizer, epoch, metrics)

        # At start of training
        resumed, start_epoch = manager.resume(model, optimizer)
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        start_epoch: int = 0,
        save_interval: int = 1,
    ):
        """Initialize resume manager.

        Args:
            checkpoint_dir: Directory for checkpoints
            start_epoch: Starting epoch (for new training)
            save_interval: Save every N epochs
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.start_epoch = start_epoch
        self.save_interval = save_interval

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Save checkpoint if needed.

        Args:
            model: PyTorch model
            optimizer: Optional optimizer
            epoch: Current epoch
            metrics: Optional metrics
        """
        from geofm.checkpoint.model_saver import ModelSaver

        # Save latest
        latest_path = self.checkpoint_dir / "latest.pt"

        ModelSaver.save_checkpoint(
            model=model,
            optimizer=optimizer,
            path=latest_path,
            epoch=epoch,
            metrics=metrics,
        )

        # Save epoch checkpoint
        if epoch % self.save_interval == 0:
            epoch_path = self.checkpoint_dir / f"epoch_{epoch}.pt"

            ModelSaver.save_checkpoint(
                model=model,
                optimizer=optimizer,
                path=epoch_path,
                epoch=epoch,
                metrics=metrics,
            )

        # Save best
        if metrics and "loss" in metrics:
            best_path = self.checkpoint_dir / "best.pt"
            current_best = None

            if best_path.exists():
                best_ckpt = torch.load(best_path)
                current_best = best_ckpt.get("metrics", {}).get("loss")

            if current_best is None or metrics["loss"] < current_best:
                ModelSaver.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    path=best_path,
                    epoch=epoch,
                    metrics=metrics,
                    extra={"is_best": True},
                )

    def resume(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        map_location: str = "cpu",
    ) -> Tuple[bool, int]:
        """Resume training.

        Args:
            model: PyTorch model
            optimizer: Optional optimizer
            map_location: Device to load to

        Returns:
            Tuple of (resumed, start_epoch)
        """
        return auto_resume(
            model=model,
            checkpoint_dir=self.checkpoint_dir,
            optimizer=optimizer,
            map_location=map_location,
        )