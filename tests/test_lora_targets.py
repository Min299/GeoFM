"""tests/test_lora_targets.py

Tests for LoRA targets.
"""
import pytest


class TestLoRATargets:
    """Test LoRATargets class."""

    def test_default_targets(self):
        """Default targets should have correct values."""
        from geofm.models.peft.lora_targets import LoRATargets

        targets = LoRATargets()

        assert targets.qkv == "attn.qkv"
        assert targets.proj == "attn.proj"
        assert targets.fc1 == "mlp.fc1"
        assert targets.fc2 == "mlp.fc2"

    def test_custom_targets(self):
        """Custom targets should store values."""
        from geofm.models.peft.lora_targets import LoRATargets

        targets = LoRATargets(
            qkv="attention.qkv",
            proj="attention.proj",
        )

        assert targets.qkv == "attention.qkv"
        assert targets.proj == "attention.proj"

    def test_default_lora_targets(self):
        """DEFAULT_LORA_TARGETS should exist."""
        from geofm.models.peft.lora_targets import DEFAULT_LORA_TARGETS

        assert DEFAULT_LORA_TARGETS is not None
        assert DEFAULT_LORA_TARGETS.qkv == "attn.qkv"

    def test_terramind_targets(self):
        """TERRAMIND_TARGETS should exist."""
        from geofm.models.peft.lora_targets import TERRAMIND_TARGETS

        assert TERRAMIND_TARGETS is not None
        assert "attn" in TERRAMIND_TARGETS.qkv

    def test_prithvi_targets(self):
        """PRITHVI_TARGETS should exist."""
        from geofm.models.peft.lora_targets import PRITHVI_TARGETS

        assert PRITHVI_TARGETS is not None
        assert "attention" in PRITHVI_TARGETS.qkv

    def test_get_lora_targets(self):
        """get_lora_targets should return correct targets."""
        from geofm.models.peft.lora_targets import get_lora_targets

        targets = get_lora_targets("terramind")

        assert targets.qkv == "attn.qkv"

    def test_get_lora_targets_unknown(self):
        """Unknown model should return defaults."""
        from geofm.models.peft.lora_targets import get_lora_targets, DEFAULT_LORA_TARGETS

        targets = get_lora_targets("unknown_model")

        assert targets.qkv == DEFAULT_LORA_TARGETS.qkv

    def test_get_target_names(self):
        """get_target_names should return list."""
        from geofm.models.peft.lora_targets import get_target_names, TERRAMIND_TARGETS

        names = get_target_names(TERRAMIND_TARGETS, include_mlp=False)

        assert len(names) == 2
        assert "attn.qkv" in names
        assert "attn.proj" in names

    def test_get_target_names_with_mlp(self):
        """get_target_names with MLP should include fc layers."""
        from geofm.models.peft.lora_targets import get_target_names, TERRAMIND_TARGETS

        names = get_target_names(TERRAMIND_TARGETS, include_mlp=True)

        # TERRAMIND has qkv, proj, fc1, fc2, fc3 (fc3 has default value)
        assert len(names) == 5
        assert "mlp.fc1" in names
        assert "mlp.fc2" in names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])