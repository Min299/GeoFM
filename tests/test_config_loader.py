"""tests/test_config_loader.py

Tests for config loader.
"""
import pytest
from geofm.config.experiment_config import ExperimentConfig


class TestExperimentConfig:
    """Test ExperimentConfig class."""

    def test_config_object(self):
        """Config should create with required fields."""
        cfg = ExperimentConfig(
            experiment_name="test",
            task="flood",
            model_type="shared",
            adapter_type="feature",
        )

        assert cfg.experiment_name == "test"
        assert cfg.task == "flood"
        assert cfg.model_type == "shared"
        assert cfg.adapter_type == "feature"

    def test_config_defaults(self):
        """Config should have correct defaults."""
        cfg = ExperimentConfig(
            experiment_name="test",
            task="flood",
            model_type="shared",
            adapter_type="feature",
        )

        assert cfg.epochs == 10
        assert cfg.batch_size == 8
        assert cfg.learning_rate == 1e-4
        assert cfg.seed == 42

    def test_to_dict(self):
        """to_dict should return dictionary."""
        cfg = ExperimentConfig(
            experiment_name="test",
            task="flood",
            model_type="shared",
            adapter_type="feature",
        )

        d = cfg.to_dict()

        assert d["experiment_name"] == "test"
        assert d["task"] == "flood"

    def test_from_dict(self):
        """from_dict should create config."""
        data = {
            "experiment_name": "test",
            "task": "flood",
            "model_type": "shared",
            "adapter_type": "feature",
        }

        cfg = ExperimentConfig.from_dict(data)

        assert cfg.experiment_name == "test"

    def test_update(self):
        """update should modify config."""
        cfg = ExperimentConfig(
            experiment_name="test",
            task="flood",
            model_type="shared",
            adapter_type="feature",
        )

        cfg.update(epochs=20, learning_rate=1e-3)

        assert cfg.epochs == 20
        assert cfg.learning_rate == 1e-3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])