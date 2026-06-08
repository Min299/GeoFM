"""geofm.checkpoint.model_loader

Model loading utilities.

Handles loading PyTorch models and checkpoints.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Union, Optional
import torch


class ModelLoader:
    """Utility for loading PyTorch models.

    Usage:
        ModelLoader.load(model, "checkpoints/model.pt")
        checkpoint = ModelLoader.load_checkpoint("checkpoints/ckpt.pt")
    """

    @staticmethod
    def load(
        model: torch.nn.Module,
        path: Union[str, Path],
        map_location: str = "cpu",
        strict: bool = True,
    ) -> torch.nn.Module:
        """Load model state dict.

        Args:
            model: PyTorch model
            path: Path to checkpoint
            map_location: Device to load to
            strict: Whether to strictly enforce key matching

        Returns:
            Model with loaded weights
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        state = torch.load(path, map_location=map_location)

        model.load_state_dict(state, strict=strict)

        return model

    @staticmethod
    def load_checkpoint(
        path: Union[str, Path],
        map_location: str = "cpu",
    ) -> Dict[str, Any]:
        """Load checkpoint without model.

        Args:
            path: Path to checkpoint
            map_location: Device to load to

        Returns:
            Checkpoint dictionary
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=map_location)

        return checkpoint

    @staticmethod
    def load_partial(
        model: torch.nn.Module,
        path: Union[str, Path],
        map_location: str = "cpu",
    ) -> torch.nn.Module:
        """Load checkpoint with partial matching.

        Useful when loading from a model with slightly different architecture.

        Args:
            model: PyTorch model
            path: Path to checkpoint
            map_location: Device to load to

        Returns:
            Model with loaded weights
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        state = torch.load(path, map_location=map_location)

        # Filter state dict to only include matching keys
        model_state = model.state_dict()
        filtered_state = {}

        for key, value in state.items():
            if key in model_state:
                if model_state[key].shape == value.shape:
                    filtered_state[key] = value

        model.load_state_dict(filtered_state, strict=False)

        return model

    @staticmethod
    def get_checkpoint_info(
        path: Union[str, Path],
        map_location: str = "cpu",
    ) -> Dict[str, Any]:
        """Get information about a checkpoint without loading full model.

        Args:
            path: Path to checkpoint
            map_location: Device to load to

        Returns:
            Dictionary with checkpoint info
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=map_location)

        info = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
        }

        if "epoch" in checkpoint:
            info["epoch"] = checkpoint["epoch"]

        if "metrics" in checkpoint:
            info["metrics"] = checkpoint["metrics"]

        if "model_state_dict" in checkpoint:
            info["num_parameters"] = len(checkpoint["model_state_dict"])

        return info


def load_model(
    model: torch.nn.Module,
    path: Union[str, Path],
    map_location: str = "cpu",
) -> torch.nn.Module:
    """Convenience function to load model.

    Args:
        model: PyTorch model
        path: Path to checkpoint
        map_location: Device to load to

    Returns:
        Model with loaded weights
    """
    return ModelLoader.load(model, path, map_location)


def load_checkpoint(
    path: Union[str, Path],
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """Convenience function to load checkpoint.

    Args:
        path: Path to checkpoint
        map_location: Device to load to

    Returns:
        Checkpoint dictionary
    """
    return ModelLoader.load_checkpoint(path, map_location)