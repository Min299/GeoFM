"""tests/test_lora_config.py

Tests for LoRA configuration.
"""
import pytest


class TestLoRAConfig:
    """Test LoRAConfig class."""

    def test_default_config(self):
        """Default config should have correct values."""
        from geofm.models.peft.lora_config import LoRAConfig

        config = LoRAConfig()

        assert config.rank == 16
        assert config.alpha == 32
        assert config.dropout == 0.05
        assert config.target_qkv is True
        assert config.target_proj is True
        assert config.target_mlp is False

    def test_custom_config(self):
        """Custom config should store values."""
        from geofm.models.peft.lora_config import LoRAConfig

        config = LoRAConfig(rank=8, alpha=16, dropout=0.1)

        assert config.rank == 8
        assert config.alpha == 16
        assert config.dropout == 0.1

    def test_invalid_rank(self):
        """Invalid rank should raise."""
        from geofm.models.peft.lora_config import LoRAConfig

        with pytest.raises(ValueError, match="rank must be positive"):
            LoRAConfig(rank=0)

    def test_invalid_alpha(self):
        """Invalid alpha should raise."""
        from geofm.models.peft.lora_config import LoRAConfig

        with pytest.raises(ValueError, match="alpha must be positive"):
            LoRAConfig(alpha=0)

    def test_invalid_dropout(self):
        """Invalid dropout should raise."""
        from geofm.models.peft.lora_config import LoRAConfig

        with pytest.raises(ValueError, match="dropout must be in"):
            LoRAConfig(dropout=1.5)

    def test_scale_property(self):
        """scale should return alpha / rank."""
        from geofm.models.peft.lora_config import LoRAConfig

        config = LoRAConfig(rank=16, alpha=32)

        assert config.scale == 2.0

    def test_get_rank_dict(self):
        """get_rank_dict should return qkv ranks."""
        from geofm.models.peft.lora_config import LoRAConfig

        config = LoRAConfig(rank=16)

        rank_dict = config.get_rank_dict()

        assert rank_dict["q"] == 16
        assert rank_dict["k"] == 16
        assert rank_dict["v"] == 16

    def test_presets(self):
        """Preset configs should exist."""
        from geofm.models.peft.lora_config import LORA_CONFIGS

        assert "tiny" in LORA_CONFIGS
        assert "small" in LORA_CONFIGS
        assert "base" in LORA_CONFIGS
        assert "large" in LORA_CONFIGS

    def test_get_lora_config(self):
        """get_lora_config should return preset."""
        from geofm.models.peft.lora_config import get_lora_config

        config = get_lora_config("base")

        assert config.rank == 16
        assert config.alpha == 32

    def test_get_lora_config_unknown(self):
        """Unknown config should raise."""
        from geofm.models.peft.lora_config import get_lora_config

        with pytest.raises(ValueError, match="Unknown LoRA config"):
            get_lora_config("unknown")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])