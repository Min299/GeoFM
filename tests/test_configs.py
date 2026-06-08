"""tests/test_configs.py

Tests for experiment config files.
"""
import pytest
from pathlib import Path

import yaml


class TestExperimentConfigs:
    """Test experiment configuration files."""

    @pytest.fixture
    def configs_dir(self):
        """Get configs/experiments directory."""
        return Path(__file__).parent.parent / "configs" / "experiments"

    def test_configs_exist(self, configs_dir):
        """All config files should exist."""
        expected_configs = [
            "flood_feature.yaml",
            "flood_lora.yaml",
            "flood_hybrid.yaml",
            "flood_fullft.yaml",
        ]

        for cfg in expected_configs:
            assert (configs_dir / cfg).exists(), f"Missing config: {cfg}"

    def test_flood_feature_valid(self, configs_dir):
        """flood_feature.yaml should be valid YAML."""
        config_path = configs_dir / "flood_feature.yaml"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert config["experiment_name"] == "flood_feature"
        assert config["task"] == "flood"
        assert config["adapter_type"] == "feature"
        assert config["backbone"]["frozen"] is True

    def test_flood_lora_valid(self, configs_dir):
        """flood_lora.yaml should be valid YAML."""
        config_path = configs_dir / "flood_lora.yaml"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert config["experiment_name"] == "flood_lora"
        assert config["task"] == "flood"
        assert config["adapter_type"] == "lora"
        assert "lora" in config
        assert config["lora"]["rank"] == 16

    def test_flood_hybrid_valid(self, configs_dir):
        """flood_hybrid.yaml should be valid YAML."""
        config_path = configs_dir / "flood_hybrid.yaml"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert config["experiment_name"] == "flood_hybrid"
        assert config["adapter_type"] == "hybrid"
        assert "lora" in config
        assert "feature_adapter" in config

    def test_flood_fullft_valid(self, configs_dir):
        """flood_fullft.yaml should be valid YAML."""
        config_path = configs_dir / "flood_fullft.yaml"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert config["experiment_name"] == "flood_fullft"
        assert config["adapter_type"] == "full_ft"
        assert config["backbone"]["frozen"] is False

    def test_all_configs_have_training(self, configs_dir):
        """All configs should have training section."""
        config_files = configs_dir.glob("flood_*.yaml")

        for config_path in config_files:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            assert "training" in config
            assert "batch_size" in config["training"]
            assert "epochs" in config["training"]
            assert "learning_rate" in config["training"]

    def test_all_configs_have_evaluation(self, configs_dir):
        """All configs should have evaluation section."""
        config_files = configs_dir.glob("flood_*.yaml")

        for config_path in config_files:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            assert "evaluation" in config
            assert "metrics" in config["evaluation"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])