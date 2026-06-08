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
    MultiTaskReport as BenchmarkMultiTaskReport,
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

# Phase 5D - Multi-Task Learning
from geofm.evaluation.multitask_metrics import (
    MultiTaskMetrics as TaskMultiTaskMetrics,
    TaskMetrics as MultiTaskTaskMetrics,
    MetricTracker,
    AggregatedMetrics,
)
from geofm.evaluation.multitask_report import (
    MultiTaskReport,
    ComparisonReport,
    TrainingProgressReport,
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
    "BenchmarkMultiTaskReport",
    "ExperimentComparison",
    "ResultsWriter",
    "CSVResultsWriter",
    "TensorBoardWriter",
    "Leaderboard",
    "LeaderboardEntry",
    "MultiMetricLeaderboard",
    # Phase 5D - Multi-Task Learning
    "TaskMultiTaskMetrics",
    "MetricTracker",
    "AggregatedMetrics",
    "MultiTaskReport",
    "ComparisonReport",
    "TrainingProgressReport",
]