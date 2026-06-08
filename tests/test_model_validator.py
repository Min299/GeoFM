"""tests/test_model_validator.py

Tests for ModelValidator.
"""
import pytest


class DummyModel:
    """Dummy model for testing."""

    def __init__(self):
        self.backbone = object()
        self.adapter_bank = {}
        self.decoder_bank = {"flood": object(), "burn": object()}

    def forward(self, batch, task_name):
        return {}


class TestModelValidator:
    """Test ModelValidator class."""

    def test_validate_model_success(self):
        """Valid model should pass."""
        from geofm.integration import ModelValidator

        model = DummyModel()

        # Should not raise
        ModelValidator.validate_model(model)

    def test_validate_model_missing_backbone(self):
        """Missing backbone should raise."""
        from geofm.integration import ModelValidator

        class IncompleteModel:
            adapter_bank = {}
            decoder_bank = {}

        with pytest.raises(ValueError, match="Model missing required component: backbone"):
            ModelValidator.validate_model(IncompleteModel())

    def test_validate_model_missing_adapter_bank(self):
        """Missing adapter_bank should raise."""
        from geofm.integration import ModelValidator

        class IncompleteModel:
            backbone = object()
            decoder_bank = {}

        with pytest.raises(ValueError, match="Model missing required component: adapter_bank"):
            ModelValidator.validate_model(IncompleteModel())

    def test_validate_model_missing_decoder_bank(self):
        """Missing decoder_bank should raise."""
        from geofm.integration import ModelValidator

        class IncompleteModel:
            backbone = object()
            adapter_bank = {}

        with pytest.raises(ValueError, match="Model missing required component: decoder_bank"):
            ModelValidator.validate_model(IncompleteModel())

    def test_validate_task_success(self):
        """Registered task should pass."""
        from geofm.integration import ModelValidator

        model = DummyModel()

        ModelValidator.validate_task(model, "flood")

    def test_validate_task_not_registered(self):
        """Unregistered task should raise."""
        from geofm.integration import ModelValidator

        model = DummyModel()

        with pytest.raises(ValueError, match="not registered"):
            ModelValidator.validate_task(model, "lulc")

    def test_validate_forward_method(self):
        """Model with forward should pass."""
        from geofm.integration import ModelValidator

        model = DummyModel()

        ModelValidator.validate_forward_method(model)

    def test_validate_all(self):
        """validate_all should run all checks."""
        from geofm.integration import ModelValidator

        model = DummyModel()

        ModelValidator.validate_all(model, task="flood")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])