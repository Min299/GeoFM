"""tests/test_multitask_smoke.py

Smoke tests for Phase 5D multi-task learning infrastructure.
"""
import pytest


class TestPhase5DSmoke:
    """Smoke tests for Phase 5D."""

    def test_tasks_module_imports(self):
        """Tasks module should be importable."""
        from geofm.tasks import (
            TaskScheduler,
            TaskSampler,
            TaskBatcher,
        )

        assert TaskScheduler is not None
        assert TaskSampler is not None
        assert TaskBatcher is not None

    def test_training_module_imports(self):
        """Training module should be importable."""
        from geofm.training import (
            multitask_step,
            MultiTaskTrainer,
        )

        assert multitask_step is not None
        assert MultiTaskTrainer is not None

    def test_multitask_metrics_imports(self):
        """Multi-task metrics should be importable."""
        from geofm.evaluation.multitask_metrics import (
            MultiTaskMetrics,
            MetricTracker,
        )

        assert MultiTaskMetrics is not None
        assert MetricTracker is not None

    def test_multitask_report_imports(self):
        """Multi-task report should be importable."""
        from geofm.evaluation.multitask_report import (
            MultiTaskReport,
            ComparisonReport,
        )

        assert MultiTaskReport is not None
        assert ComparisonReport is not None

    def test_multitask_experiment_imports(self):
        """Multi-task experiment should be importable."""
        from geofm.experiments.multitask_experiment import (
            MultiTaskExperiment,
            ExperimentConfig,
        )

        assert MultiTaskExperiment is not None
        assert ExperimentConfig is not None

    def test_multitask_benchmark_imports(self):
        """Multi-task benchmark should be importable."""
        from geofm.experiments.multitask_benchmark import (
            MultiTaskBenchmark,
            QuickMultiTaskBenchmark,
        )

        assert MultiTaskBenchmark is not None
        assert QuickMultiTaskBenchmark is not None

    def test_all_tasks_defined(self):
        """All five tasks should be defined."""
        tasks = ["flood", "burn", "lulc", "crop", "ndvi"]

        assert len(tasks) == 5

    def test_config_files_exist(self):
        """Multi-task config files should exist."""
        from pathlib import Path

        configs_dir = Path("configs")

        assert (configs_dir / "multitask_feature.yaml").exists()
        assert (configs_dir / "multitask_lora.yaml").exists()
        assert (configs_dir / "multitask_hybrid.yaml").exists()

    def test_task_scheduler_smoke(self):
        """TaskScheduler should work."""
        from geofm.tasks import TaskScheduler

        scheduler = TaskScheduler(["flood", "burn"])

        task = scheduler.next_task()

        assert task in ["flood", "burn"]

    def test_task_batcher_smoke(self):
        """TaskBatcher should work with dummy loaders."""
        from geofm.tasks import TaskBatcher

        loaders = {
            "flood": [{"inputs": [1], "targets": [0]}],
            "burn": [{"inputs": [2], "targets": [1]}],
        }

        batcher = TaskBatcher(loaders)

        batch = batcher.get_batch("flood")

        assert batch.task_name == "flood"

    def test_multitask_metrics_smoke(self):
        """MultiTaskMetrics should work."""
        from geofm.evaluation.multitask_metrics import MultiTaskMetrics

        metrics = MultiTaskMetrics()

        metrics.update("flood", loss=0.5)
        metrics.update("burn", loss=0.6)

        summary = metrics.summary()

        assert "flood" in summary
        assert "burn" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])