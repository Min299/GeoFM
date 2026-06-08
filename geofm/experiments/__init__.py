"""geofm.experiments

Experiment tracking and running for GeoFM.
"""
from geofm.experiments.experiment import (
    ExperimentConfig,
    ExperimentResult,
    create_experiment,
)
from geofm.experiments.experiment_builder import (
    ExperimentBuilder,
    FeatureAdapterModel,
    build_experiment_model,
)
from geofm.experiments.experiment_runner import ExperimentRunner
from geofm.experiments.registry import (
    ExperimentRegistry,
    get_registry,
)

__all__ = [
    # Config
    "ExperimentConfig",
    "ExperimentResult",
    "create_experiment",
    # Builder
    "ExperimentBuilder",
    "FeatureAdapterModel",
    "build_experiment_model",
    # Runner
    "ExperimentRunner",
    # Registry
    "ExperimentRegistry",
    "get_registry",
]