"""geofm.evaluation

Evaluation metrics for GeoFM tasks.
"""
from geofm.evaluation.segmentation_metrics import SegmentationMetrics
from geofm.evaluation.classification_metrics import ClassificationMetrics
from geofm.evaluation.regression_metrics import RegressionMetrics
from geofm.evaluation.metric_registry import (
    get_metrics,
    get_primary_metric,
    is_segmentation_task,
    is_classification_task,
    is_regression_task,
)

# Phase 5C - Benchmark Evaluation Infrastructure
from geofm.evaluation.benchmark_metrics import (
    BenchmarkMetrics,
    TaskMetrics,
    ExperimentMetrics,
)
from geofm.evaluation.benchmark_report import (
    BenchmarkReport,
    MultiTaskReport,
    ExperimentComparison,
)
from geofm.evaluation.results_writer import (
    ResultsWriter,
    CSVResultsWriter,
    TensorBoardWriter,
)
from geofm.evaluation.leaderboard import (
    Leaderboard,
    LeaderboardEntry,
    MultiMetricLeaderboard,
)

__all__ = [
    # Metrics
    "SegmentationMetrics",
    "ClassificationMetrics",
    "RegressionMetrics",
    # Registry
    "get_metrics",
    "get_primary_metric",
    "is_segmentation_task",
    "is_classification_task",
    "is_regression_task",
    # Phase 5C - Benchmark Evaluation
    "BenchmarkMetrics",
    "TaskMetrics",
    "ExperimentMetrics",
    "BenchmarkReport",
    "MultiTaskReport",
    "ExperimentComparison",
    "ResultsWriter",
    "CSVResultsWriter",
    "TensorBoardWriter",
    "Leaderboard",
    "LeaderboardEntry",
    "MultiMetricLeaderboard",
]