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
from geofm.experiments.experiment_tracker import (
    ExperimentTracker,
    ExperimentRecord,
)
from geofm.experiments.result_manager import (
    ResultManager,
)
from geofm.experiments.experiment_runner import ExperimentRunner
from geofm.experiments.registry import (
    ExperimentRegistry,
    get_registry,
)

# Phase 5C - Adaptation Benchmark Infrastructure
from geofm.experiments.experiment_registry import (
    EXPERIMENTS,
    get_experiment,
    list_experiments,
    get_adapter_type,
    should_freeze_backbone,
)
from geofm.experiments.benchmark_config import (
    BenchmarkConfig,
    BenchmarkSuiteConfig,
)
from geofm.experiments.adaptation_benchmark import (
    AdaptationBenchmark,
    MultiTaskBenchmark,
)
from geofm.experiments.benchmark_runner import (
    BenchmarkRunner,
    SuiteRunner,
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
    # Tracker
    "ExperimentTracker",
    "ExperimentRecord",
    # Result Manager
    "ResultManager",
    # Runner
    "ExperimentRunner",
    # Registry
    "ExperimentRegistry",
    "get_registry",
    # Phase 5C - Adaptation Benchmark
    "EXPERIMENTS",
    "get_experiment",
    "list_experiments",
    "get_adapter_type",
    "should_freeze_backbone",
    "BenchmarkConfig",
    "BenchmarkSuiteConfig",
    "AdaptationBenchmark",
    "MultiTaskBenchmark",
    "BenchmarkRunner",
    "SuiteRunner",
]