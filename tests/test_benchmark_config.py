"""tests/test_benchmark_config.py

Tests for benchmark config.
"""
import pytest


class TestBenchmarkConfig:
    """Test benchmark config."""

    def test_config_init(self):
        """Config should initialize with defaults."""
        from geofm.experiments.benchmark_config import BenchmarkConfig

        config = BenchmarkConfig(
            experiment_name="feature",
            task="flood",
        )

        assert config.experiment_name == "feature"
        assert config.task == "flood"
        assert config.epochs == 10
        assert config.batch_size == 8
        assert config.learning_rate == 1e-4

    def test_config_custom(self):
        """Config should accept custom values."""
        from geofm.experiments.benchmark_config import BenchmarkConfig

        config = BenchmarkConfig(
            experiment_name="lora",
            task="flood",
            epochs=20,
            batch_size=16,
            learning_rate=1e-3,
        )

        assert config.epochs == 20
        assert config.batch_size == 16
        assert config.learning_rate == 1e-3

    def test_to_dict(self):
        """to_dict should return dictionary."""
        from geofm.experiments.benchmark_config import BenchmarkConfig

        config = BenchmarkConfig(
            experiment_name="feature",
            task="flood",
        )

        d = config.to_dict()

        assert d["experiment_name"] == "feature"
        assert d["task"] == "flood"
        assert "epochs" in d

    def test_from_dict(self):
        """from_dict should create config."""
        from geofm.experiments.benchmark_config import BenchmarkConfig

        data = {
            "experiment_name": "lora",
            "task": "flood",
            "epochs": 15,
        }

        config = BenchmarkConfig.from_dict(data)

        assert config.experiment_name == "lora"
        assert config.epochs == 15


class TestBenchmarkSuiteConfig:
    """Test benchmark suite config."""

    def test_suite_init(self):
        """Suite config should initialize."""
        from geofm.experiments.benchmark_config import BenchmarkSuiteConfig

        config = BenchmarkSuiteConfig(
            experiments=["feature", "lora"],
            task="flood",
        )

        assert len(config.experiments) == 2
        assert config.task == "flood"

    def test_get_experiment_config(self):
        """get_experiment_config should return correct config."""
        from geofm.experiments.benchmark_config import BenchmarkSuiteConfig

        suite = BenchmarkSuiteConfig(
            experiments=["feature", "lora"],
            task="flood",
            epochs=20,
            batch_size=16,
        )

        exp_config = suite.get_experiment_config("feature")

        assert exp_config.experiment_name == "feature"
        assert exp_config.task == "flood"
        assert exp_config.epochs == 20
        assert exp_config.batch_size == 16


if __name__ == "__main__":
    pytest.main([__file__, "-v"])