"""geofm.config

Configuration management for GeoFM.

Provides configuration loading, validation, and management.
"""
from geofm.config.experiment_config import (
    ExperimentConfig,
    MultiTaskExperimentConfig,
    TrainingConfig,
    ModelConfig,
)

from geofm.config.config_loader import (
    ConfigLoader,
    ConfigMerger,
)

from geofm.config.config_validator import (
    ConfigValidator,
    validate_config,
)


__all__ = [
    # Config dataclasses
    "ExperimentConfig",
    "MultiTaskExperimentConfig",
    "TrainingConfig",
    "ModelConfig",
    # Loader
    "ConfigLoader",
    "ConfigMerger",
    # Validator
    "ConfigValidator",
    "validate_config",
]