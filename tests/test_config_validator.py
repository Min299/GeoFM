"""tests/test_config_validator.py

Tests for config validator.
"""
import pytest
from geofm.config.experiment_config import ExperimentConfig
from geofm.config.config_validator import ConfigValidator, validate_config


class TestConfigValidator:
    """Test ConfigValidator class."""

    def test_validate_valid(self):
        """validate should return True for valid config."""
        cfg = ExperimentConfig(
            experiment_name="test",
            task="flood",
            model_type="shared",
            adapter_type="feature",
        )

        assert ConfigValidator.validate(cfg) is True

    def test_validate_missing_fields(self):
        """validate should raise for missing fields."""
        # Create a minimal dict to test validation
        data = {
            "experiment_name": "test",
            "task": "flood",
            "model_type": "shared",
            # adapter_type missing
        }

        with pytest.raises(ValueError) as exc:
            ConfigValidator.validate(data)

        assert "Missing required fields" in str(exc.value)

    def test_validate_invalid_task(self):
        """validate should reject invalid task."""
        cfg = ExperimentConfig(
            experiment_name="test",
            task="invalid_task",
            model_type="shared",
            adapter_type="feature",
        )

        with pytest.raises(ValueError) as exc:
            ConfigValidator.validate(cfg)

        assert "Invalid task" in str(exc.value)

    def test_validate_invalid_model_type(self):
        """validate should reject invalid model type."""
        cfg = ExperimentConfig(
            experiment_name="test",
            task="flood",
            model_type="invalid",
            adapter_type="feature",
        )

        with pytest.raises(ValueError) as exc:
            ConfigValidator.validate(cfg)

        assert "Invalid model_type" in str(exc.value)

    def test_validate_invalid_adapter_type(self):
        """validate should reject invalid adapter type."""
        cfg = ExperimentConfig(
            experiment_name="test",
            task="flood",
            model_type="shared",
            adapter_type="invalid",
        )

        with pytest.raises(ValueError) as exc:
            ConfigValidator.validate(cfg)

        assert "Invalid adapter_type" in str(exc.value)

    def test_validate_invalid_epochs(self):
        """validate should reject invalid epochs."""
        cfg = ExperimentConfig(
            experiment_name="test",
            task="flood",
            model_type="shared",
            adapter_type="feature",
            epochs=-1,
        )

        with pytest.raises(ValueError) as exc:
            ConfigValidator.validate(cfg)

        assert "epochs" in str(exc.value)

    def test_validate_invalid_learning_rate(self):
        """validate should reject invalid learning rate."""
        cfg = ExperimentConfig(
            experiment_name="test",
            task="flood",
            model_type="shared",
            adapter_type="feature",
            learning_rate=0,
        )

        with pytest.raises(ValueError) as exc:
            ConfigValidator.validate(cfg)

        assert "learning_rate" in str(exc.value)

    def test_validate_dict(self):
        """validate should work with dict input."""
        data = {
            "experiment_name": "test",
            "task": "flood",
            "model_type": "shared",
            "adapter_type": "feature",
        }

        assert ConfigValidator.validate(data) is True

    def test_get_validation_report(self):
        """get_validation_report should return report."""
        cfg = ExperimentConfig(
            experiment_name="test",
            task="flood",
            model_type="shared",
            adapter_type="feature",
        )

        report = ConfigValidator.get_validation_report(cfg)

        assert report["valid"] is True
        assert len(report["errors"]) == 0

    def test_validate_config_function(self):
        """validate_config convenience function should work."""
        cfg = ExperimentConfig(
            experiment_name="test",
            task="flood",
            model_type="shared",
            adapter_type="feature",
        )

        assert validate_config(cfg) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])