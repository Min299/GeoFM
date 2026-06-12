"""geofm.trainers.checkpoint_manager

Checkpoint save/load utilities.
"""
import torch
from pathlib import Path
from typing import Optional, Dict, Any


class CheckpointManager:
    """Manager for model checkpoints.

    Usage:
        # Save
        CheckpointManager.save("model.pt", model, optimizer)

        # Load
        model = CheckpointManager.load("model.pt", model, optimizer)
    """

    @staticmethod
    def save(
        path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save model checkpoint.

        Args:
            path: Save path
            model: Model to save
            optimizer: Optimizer state (optional)
            scheduler: Scheduler state (optional)
            epoch: Current epoch (optional)
            metrics: Training metrics (optional)
            extra: Extra data to save (optional)
        """
        payload = {
            "model_state_dict": model.state_dict(),
        }

        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()

        if scheduler is not None:
            payload["scheduler_state_dict"] = scheduler.state_dict()

        if epoch is not None:
            payload["epoch"] = epoch

        if metrics is not None:
            payload["metrics"] = metrics

        if extra is not None:
            payload["extra"] = extra

        # Create directory
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        torch.save(payload, path)

    @staticmethod
    def load(
        path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        map_location: str = "cpu",
    ) -> Dict[str, Any]:
        """Load model checkpoint.

        Args:
            path: Checkpoint path
            model: Model to load into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            map_location: Device to map to

        Returns:
            Checkpoint metadata
        """
        ckpt = torch.load(path, map_location=map_location)

        model.load_state_dict(ckpt["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        # Return metadata
        metadata = {
            k: v for k, v in ckpt.items()
            if k not in ["model_state_dict", "optimizer_state_dict", "scheduler_state_dict"]
        }

        return metadata

    @staticmethod
    def save_best(
        path: str,
        model: torch.nn.Module,
        metrics: Dict[str, float],
        current_metric: float,
        mode: str = "min",
    ) -> bool:
        """Save checkpoint only if it's the best so far.

        Args:
            path: Save path
            model: Model to save
            metrics: Current metrics
            current_metric: Metric value to compare
            mode: "min" or "max"

        Returns:
            True if saved as best, False otherwise
        """
        best_path = path.replace(".pt", "_best.pt")

        if Path(best_path).exists():
            best_ckpt = torch.load(best_path, map_location="cpu")
            best_value = best_ckpt.get("metrics", {}).get("loss", float("inf"))
        else:
            best_value = float("inf") if mode == "min" else float("-inf")

        is_best = (
            (mode == "min" and current_metric < best_value) or
            (mode == "max" and current_metric > best_value)
        )

        if is_best:
            CheckpointManager.save(
                best_path,
                model,
                metrics=metrics,
            )
            return True

        return False

    @staticmethod
    def list_checkpoints(
        directory: str,
        pattern: str = "*.pt",
    ) -> list:
        """List all checkpoints in a directory.

        Args:
            directory: Directory to search
            pattern: File pattern

        Returns:
            List of checkpoint paths
        """
        return sorted(Path(directory).glob(pattern))

    @staticmethod
    def get_latest_checkpoint(
        directory: str,
        pattern: str = "*.pt",
    ) -> Optional[str]:
        """Get the most recent checkpoint.

        Args:
            directory: Directory to search
            pattern: File pattern

        Returns:
            Path to latest checkpoint or None
        """
        checkpoints = CheckpointManager.list_checkpoints(directory, pattern)
        return str(checkpoints[-1]) if checkpoints else None