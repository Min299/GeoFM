"""tests/test_lora_injector.py

Tests for LoRA injector.
"""
import pytest
import torch.nn as nn


class DummyAttn(nn.Module):
    """Dummy attention module for testing."""

    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(768, 2304)
        self.proj = nn.Linear(768, 768)


class DummyBlock(nn.Module):
    """Dummy transformer block for testing."""

    def __init__(self):
        super().__init__()
        self.attn = DummyAttn()
        self.mlp = nn.Sequential(
            nn.Linear(768, 3072),
            nn.GELU(),
            nn.Linear(3072, 768),
        )


class DummyEncoder(nn.Module):
    """Dummy encoder with multiple blocks."""

    def __init__(self, num_blocks=12):
        super().__init__()
        self.blocks = nn.ModuleList([DummyBlock() for _ in range(num_blocks)])


class TestLoRAInjector:
    """Test LoRAInjector class."""

    def test_injector_init(self):
        """Injector should initialize with correct defaults."""
        from geofm.models.peft.lora_injector import LoRAInjector

        injector = LoRAInjector()

        assert injector.rank == 16
        assert injector.alpha == 32
        assert injector.dropout == 0.05
        assert injector.target_qkv is True
        assert injector.target_proj is True

    def test_injector_custom_params(self):
        """Injector should accept custom parameters."""
        from geofm.models.peft.lora_injector import LoRAInjector

        injector = LoRAInjector(rank=8, alpha=16, dropout=0.1)

        assert injector.rank == 8
        assert injector.alpha == 16
        assert injector.dropout == 0.1

    def test_is_attention_target_qkv(self):
        """Should identify qkv targets."""
        from geofm.models.peft.lora_injector import LoRAInjector

        injector = LoRAInjector(target_qkv=True, target_proj=False)

        assert injector._is_attention_target("blocks.0.attn.qkv") is True
        assert injector._is_attention_target("blocks.1.attn.qkv") is True

    def test_is_attention_target_proj(self):
        """Should identify proj targets."""
        from geofm.models.peft.lora_injector import LoRAInjector

        injector = LoRAInjector(target_qkv=False, target_proj=True)

        assert injector._is_attention_target("blocks.0.attn.proj") is True
        assert injector._is_attention_target("blocks.1.attn.proj") is True

    def test_is_attention_target_mlp(self):
        """Should NOT identify MLP targets (disabled by default)."""
        from geofm.models.peft.lora_injector import LoRAInjector

        injector = LoRAInjector()

        assert injector._is_attention_target("blocks.0.mlp.fc1") is False
        assert injector._is_attention_target("blocks.0.mlp.fc2") is False

    def test_inject_single_block(self):
        """Should inject LoRA into single block."""
        from geofm.models.peft.lora_injector import LoRAInjector

        model = DummyBlock()
        injector = LoRAInjector()

        replaced = injector.inject(model)

        assert len(replaced) == 2  # qkv and proj

    def test_inject_multiple_blocks(self):
        """Should inject LoRA into multiple blocks."""
        from geofm.models.peft.lora_injector import LoRAInjector

        model = DummyEncoder(num_blocks=12)
        injector = LoRAInjector()

        replaced = injector.inject(model)

        # 12 blocks * 2 targets = 24
        assert len(replaced) == 24

    def test_freeze_except_lora(self):
        """Should freeze all except LoRA params."""
        from geofm.models.peft.lora_injector import LoRAInjector

        model = DummyBlock()
        injector = LoRAInjector()
        injector.inject(model)

        # Freeze
        injector.freeze_except_lora(model)

        # Check that LoRA params are trainable
        trainable_params = [p for p in model.parameters() if p.requires_grad]

        # Should have trainable params (lora_A, lora_B)
        assert len(trainable_params) > 0

        # Verify LoRA layers exist and have trainable params
        from geofm.models.peft.lora_adapter import LoRALinear

        lora_count = sum(1 for m in model.modules() if isinstance(m, LoRALinear))
        assert lora_count == 2  # qkv and proj

    def test_count_injected(self):
        """Should count injected layers."""
        from geofm.models.peft.lora_injector import LoRAInjector

        model = DummyEncoder(num_blocks=12)
        injector = LoRAInjector()
        injector.inject(model)

        count = injector.count_injected(model)

        assert count == 24  # 12 blocks * 2 targets

    def test_get_injected_names(self):
        """Should return names of injected layers."""
        from geofm.models.peft.lora_injector import LoRAInjector

        model = DummyBlock()
        injector = LoRAInjector()
        injector.inject(model)

        names = injector.get_injected_names(model)

        assert "attn.qkv" in names
        assert "attn.proj" in names

    def test_inject_lora_into_backbone_function(self):
        """inject_lora_into_backbone should work as convenience function."""
        from geofm.models.peft.lora_injector import inject_lora_into_backbone

        model = DummyBlock()

        injector = inject_lora_into_backbone(model, rank=8, alpha=16)

        assert injector is not None
        assert injector.count_injected(model) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])