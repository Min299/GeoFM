"""tests/test_real_lora_targets.py

Tests for LoRA targets with attention-only targeting.
"""
import pytest


class TestLoRATargetsDisabledMLP:
    """Test that MLP targeting is disabled."""

    def test_default_targets_attention_only(self):
        """Default targets should only have attention targets."""
        from geofm.models.peft.lora_targets import DEFAULT_LORA_TARGETS

        assert DEFAULT_LORA_TARGETS.qkv == "attn.qkv"
        assert DEFAULT_LORA_TARGETS.proj == "attn.proj"

    def test_terramind_targets_attention_only(self):
        """TerraMind targets should only have attention targets."""
        from geofm.models.peft.lora_targets import TERRAMIND_TARGETS

        assert TERRAMIND_TARGETS.qkv == "attn.qkv"
        assert TERRAMIND_TARGETS.proj == "attn.proj"
        assert TERRAMIND_TARGETS.fc1 == ""  # Disabled
        assert TERRAMIND_TARGETS.fc2 == ""  # Disabled

    def test_prithvi_targets_attention_only(self):
        """Prithvi targets should only have attention targets."""
        from geofm.models.peft.lora_targets import PRITHVI_TARGETS

        assert PRITHVI_TARGETS.qkv == "attention.qkv"
        assert PRITHVI_TARGETS.proj == "attention.proj"
        assert PRITHVI_TARGETS.fc1 == ""  # Disabled
        assert PRITHVI_TARGETS.fc2 == ""  # Disabled

    def test_vit_targets_attention_only(self):
        """ViT targets should only have attention targets."""
        from geofm.models.peft.lora_targets import VIT_TARGETS

        assert VIT_TARGETS.qkv == "attn.qkv"
        assert VIT_TARGETS.proj == "attn.proj"
        assert VIT_TARGETS.fc1 == ""  # Disabled
        assert VIT_TARGETS.fc2 == ""  # Disabled
        assert VIT_TARGETS.fc3 == ""  # Disabled

    def test_get_target_names_attention_only(self):
        """get_target_names should return only attention targets."""
        from geofm.models.peft.lora_targets import get_target_names, TERRAMIND_TARGETS

        names = get_target_names(TERRAMIND_TARGETS, include_mlp=False)

        assert len(names) == 2
        assert "attn.qkv" in names
        assert "attn.proj" in names

    def test_get_target_names_ignores_empty_mlp(self):
        """get_target_names should ignore empty MLP targets.

        Even with include_mlp=True, empty fc1/fc2/fc3 should be filtered.
        """
        from geofm.models.peft.lora_targets import get_target_names, TERRAMIND_TARGETS

        # Even with include_mlp=True, empty strings should be filtered
        names = get_target_names(TERRAMIND_TARGETS, include_mlp=True)

        # Should only have 2 (qkv, proj) since fc1, fc2 are empty strings
        assert len(names) == 2


class TestLoRAConfigDisabledMLP:
    """Test that LoRAConfig targets MLP by default as False."""

    def test_lora_config_default_mlp_false(self):
        """LoRAConfig should default to MLP targeting off."""
        from geofm.models.peft.lora_config import LoRAConfig

        config = LoRAConfig()

        assert config.target_qkv is True
        assert config.target_proj is True
        assert config.target_mlp is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])