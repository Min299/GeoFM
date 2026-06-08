"""geofm.experiments.experiment

Experiment configuration and tracking.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime
import json
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Configuration for a GeoFM experiment.

    Attributes:
        name: Experiment name (e.g., "flood_full_ft_v1")
        task: Task type (e.g., "flood", "burn", "crop")
        adaptation: Adaptation method ("feature", "lora", "hybrid", "full_ft")
        backbone: Backbone model name
        dataset: Dataset name
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        metadata_enabled: Whether to save metadata
    """

    name: str
    task: str = "flood"
    adaptation: str = "feature"  # "feature", "lora", "hybrid", "full_ft"
    backbone: str = "terramind_base"
    dataset: str = "flood_dataset"

    # Training
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.1

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    # Paths
    output_dir: str = "outputs"
    metadata_enabled: bool = True

    # Auto-generated
    experiment_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        """Validate configuration."""
        valid_adaptations = ["feature", "lora", "hybrid", "full_ft"]
        if self.adaptation not in valid_adaptations:
            raise ValueError(
                f"Invalid adaptation: {self.adaptation}. "
                f"Must be one of {valid_adaptations}"
            )

        valid_tasks = ["flood", "burn", "lulc", "segmentation"]
        if self.task not in valid_tasks:
            raise ValueError(
                f"Invalid task: {self.task}. "
                f"Must be one of {valid_tasks}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "task": self.task,
            "adaptation": self.adaptation,
            "backbone": self.backbone,
            "dataset": self.dataset,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "output_dir": self.output_dir,
            "experiment_id": self.experiment_id,
            "created_at": self.created_at,
        }

    def save(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load config from YAML file."""
        import yaml
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


@dataclass
class ExperimentResult:
    """Results from an experiment run."""

    config: ExperimentConfig
    metrics: Dict[str, Any]
    best_epoch: int
    total_epochs: int
    duration_seconds: float
    checkpoint_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "metrics": self.metrics,
            "best_epoch": self.best_epoch,
            "total_epochs": self.total_epochs,
            "duration_seconds": self.duration_seconds,
            "checkpoint_path": self.checkpoint_path,
        }

    def save(self, path: str) -> None:
        """Save results to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def create_experiment(
    name: str,
    task: str = "flood",
    training_mode: str = "full_ft",
    **kwargs
) -> ExperimentConfig:
    """Create an experiment config with smart defaults.

    Args:
        name: Experiment name
        task: Task type
        training_mode: "full_ft" or "lora"
        **kwargs: Override defaults

    Returns:
        ExperimentConfig instance
    """
    return ExperimentConfig(
        name=name,
        task=task,
        training_mode=training_mode,
        **kwargs
    )