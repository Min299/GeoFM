"""tests/test_adaptation_benchmark.py

Tests for adaptation benchmark.
"""
import pytest
import torch
import torch.nn as nn


class DummyTrainer:
    """Dummy trainer for testing."""

    def __init__(self, model):
        self.model = model

    def train_epoch(self, loader, task=None):
        return 0.5

    def eval_epoch(self, loader, task=None):
        return 0.6


class TestAdaptationBenchmark:
    """Test adaptation benchmark."""

    def test_benchmark_init(self):
        """Benchmark should initialize."""
        from geofm.experiments.adaptation_benchmark import AdaptationBenchmark
        from geofm.experiments.benchmark_config import BenchmarkConfig

        config = BenchmarkConfig(
            experiment_name="feature",
            task="flood",
        )

        model = nn.Linear(10, 5)
        trainer = DummyTrainer(model)

        benchmark = AdaptationBenchmark(config, model, trainer)

        assert benchmark.config == config
        assert benchmark.model == model
        assert benchmark.trainer == trainer

    def test_benchmark_run(self):
        """Benchmark.run should work."""
        from geofm.experiments.adaptation_benchmark import AdaptationBenchmark
        from geofm.experiments.benchmark_config import BenchmarkConfig

        config = BenchmarkConfig(
            experiment_name="feature",
            task="flood",
            epochs=2,
        )

        model = nn.Linear(10, 5)
        trainer = DummyTrainer(model)

        benchmark = AdaptationBenchmark(config, model, trainer)

        # Dummy loader
        loader = [None, None]

        results = benchmark.run(loader)

        assert results["experiment"] == "feature"
        assert results["task"] == "flood"
        assert results["epochs"] == 2
        assert len(results["history"]) == 2

    def test_benchmark_run_single_epoch(self):
        """Benchmark.run_single_epoch should work."""
        from geofm.experiments.adaptation_benchmark import AdaptationBenchmark
        from geofm.experiments.benchmark_config import BenchmarkConfig

        config = BenchmarkConfig(
            experiment_name="feature",
            task="flood",
        )

        model = nn.Linear(10, 5)
        trainer = DummyTrainer(model)

        benchmark = AdaptationBenchmark(config, model, trainer)

        loader = [None]

        loss = benchmark.run_single_epoch(loader)

        assert loss == 0.5

    def test_get_final_loss(self):
        """get_final_loss should return last loss."""
        from geofm.experiments.adaptation_benchmark import AdaptationBenchmark
        from geofm.experiments.benchmark_config import BenchmarkConfig

        config = BenchmarkConfig(
            experiment_name="feature",
            task="flood",
            epochs=3,
        )

        model = nn.Linear(10, 5)
        trainer = DummyTrainer(model)

        benchmark = AdaptationBenchmark(config, model, trainer)
        loader = [None] * 3

        benchmark.run(loader)

        assert benchmark.get_final_loss() == 0.5

    def test_get_best_loss(self):
        """get_best_loss should return minimum loss."""
        from geofm.experiments.adaptation_benchmark import AdaptationBenchmark
        from geofm.experiments.benchmark_config import BenchmarkConfig

        config = BenchmarkConfig(
            experiment_name="feature",
            task="flood",
            epochs=3,
        )

        model = nn.Linear(10, 5)
        trainer = DummyTrainer(model)

        benchmark = AdaptationBenchmark(config, model, trainer)
        loader = [None] * 3

        benchmark.run(loader)

        assert benchmark.get_best_loss() == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])