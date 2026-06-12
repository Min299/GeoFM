"""tests/test_experiment_registry.py

Tests for experiment registry.
"""
import pytest


class TestExperimentRegistry:
    """Test experiment registry."""

    def test_registry_has_all_experiments(self):
        """Registry should have all four experiments."""
        from geofm.experiments.experiment_registry import EXPERIMENTS

        assert "feature" in EXPERIMENTS
        assert "lora" in EXPERIMENTS
        assert "hybrid" in EXPERIMENTS
        assert "fullft" in EXPERIMENTS

    def test_get_experiment(self):
        """get_experiment should return correct experiment."""
        from geofm.experiments.experiment_registry import get_experiment

        feature = get_experiment("feature")
        assert feature["adapter"] == "feature"
        assert feature["freeze_backbone"] is True

        lora = get_experiment("lora")
        assert lora["adapter"] == "lora"
        assert lora["freeze_backbone"] is True

        fullft = get_experiment("fullft")
        assert fullft["adapter"] is None
        assert fullft["freeze_backbone"] is False

    def test_get_experiment_unknown(self):
        """get_experiment should raise for unknown experiment."""
        from geofm.experiments.experiment_registry import get_experiment

        with pytest.raises(KeyError):
            get_experiment("unknown")

    def test_list_experiments(self):
        """list_experiments should return all experiment names."""
        from geofm.experiments.experiment_registry import list_experiments

        experiments = list_experiments()

        assert len(experiments) == 4
        assert "feature" in experiments
        assert "lora" in experiments
        assert "hybrid" in experiments
        assert "fullft" in experiments

    def test_get_adapter_type(self):
        """get_adapter_type should return correct adapter type."""
        from geofm.experiments.experiment_registry import get_adapter_type

        assert get_adapter_type("feature") == "feature"
        assert get_adapter_type("lora") == "lora"
        assert get_adapter_type("hybrid") == "hybrid"
        assert get_adapter_type("fullft") is None

    def test_should_freeze_backbone(self):
        """should_freeze_backbone should return correct value."""
        from geofm.experiments.experiment_registry import should_freeze_backbone

        assert should_freeze_backbone("feature") is True
        assert should_freeze_backbone("lora") is True
        assert should_freeze_backbone("hybrid") is True
        assert should_freeze_backbone("fullft") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])