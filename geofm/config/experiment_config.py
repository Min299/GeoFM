"""geofm.config.experiment_config

Experiment configuration dataclasses.

Defines the configuration structure for GeoFM experiments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class ExperimentConfig:
    """Configuration for a GeoFM experiment.

    Attributes:
        experiment_name: Name of the experiment
        task: Task to run (flood, burn, lulc, crop, ndvi)
        model_type: Type of model (shared, separate)
        adapter_type: Adapter type (feature, lora, hybrid, fullft)
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: Weight decay
        seed: Random seed
        output_dir: Output directory
        metadata_enabled: Whether to use metadata
        extra: Extra configuration dict
    """

    experiment_name: str
    task: str
    model_type: str
    adapter_type: str
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    seed: int = 42
    output_dir: str = "outputs"
    metadata_enabled: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "experiment_name": self.experiment_name,
            "task": self.task,
            "model_type": self.model_type,
            "adapter_type": self.adapter_type,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "seed": self.seed,
            "output_dir": self.output_dir,
            "metadata_enabled": self.metadata_enabled,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Create from dictionary.

        Args:
            data: Dictionary with config data

        Returns:
            ExperimentConfig instance
        """
        return cls(**data)

    def update(self, **kwargs) -> None:
        """Update config with new values.

        Args:
            **kwargs: Fields to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.extra[key] = value


@dataclass
class MultiTaskExperimentConfig(ExperimentConfig):
    """Configuration for multi-task experiments."""

    tasks: List[str] = field(default_factory=list)
    task_weights: Dict[str, float] = field(default_factory=dict)
    scheduler_type: str = "round_robin"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update({
            "tasks": self.tasks,
            "task_weights": self.task_weights,
            "scheduler_type": self.scheduler_type,
        })
        return base


@dataclass
class TrainingConfig:
    """Training-specific configuration."""

    epochs: int = 10
    steps_per_epoch: int = 100
    gradient_clip: Optional[float] = None
    accumulation_steps: int = 1
    warmup_steps: int = 0
    eval_interval: int = 1
    log_interval: int = 10
    save_interval: int = 5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "epochs": self.epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "gradient_clip": self.gradient_clip,
            "accumulation_steps": self.accumulation_steps,
            "warmup_steps": self.warmup_steps,
            "eval_interval": self.eval_interval,
            "log_interval": self.log_interval,
            "save_interval": self.save_interval,
        }


@dataclass
class ModelConfig:
    """Model-specific configuration."""

    feature_dim: int = 768
    bottleneck_dim: int = 64
    lora_rank: int = 16
    lora_alpha: int = 32
    freeze_backbone: bool = True
    use_metadata: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_dim": self.feature_dim,
            "bottleneck_dim": self.bottleneck_dim,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "freeze_backbone": self.freeze_backbone,
            "use_metadata": self.use_metadata,
        }