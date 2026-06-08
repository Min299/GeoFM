"""geofm.experiments.benchmark_config

Configuration dataclasses for benchmarking.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""

    experiment_name: str
    task: str

    # Training parameters
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01

    # Model parameters
    feature_dim: int = 768
    bottleneck_dim: int = 64
    lora_rank: int = 16
    lora_alpha: int = 32

    # Output
    output_dir: str = "results"

    # Device
    device: str = "cuda"

    # Random seed
    seed: int = 42

    def to_dict(self) -> dict:
        """Convert to dictionary.

        Returns:
            Configuration as dict
        """
        return {
            "experiment_name": self.experiment_name,
            "task": self.task,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "feature_dim": self.feature_dim,
            "bottleneck_dim": self.bottleneck_dim,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "output_dir": self.output_dir,
            "device": self.device,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, config: dict) -> "BenchmarkConfig":
        """Create from dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            BenchmarkConfig instance
        """
        return cls(**config)


@dataclass
class BenchmarkSuiteConfig:
    """Configuration for a full benchmark suite."""

    experiments: list[str] = field(default_factory=lambda: ["feature", "lora", "hybrid", "fullft"])
    task: str = "flood"

    # Training parameters
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-4

    # Output
    output_dir: str = "results"

    def get_experiment_config(self, experiment_name: str) -> BenchmarkConfig:
        """Get config for a specific experiment.

        Args:
            experiment_name: Name of experiment

        Returns:
            BenchmarkConfig for that experiment
        """
        return BenchmarkConfig(
            experiment_name=experiment_name,
            task=self.task,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            output_dir=self.output_dir,
        )