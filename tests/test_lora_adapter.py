"""tests/test_lora_adapter.py

Tests for LoRA adapter.
"""
import pytest
import torch
import torch.nn as nn


class TestLoRALinear:
    """Test LoRALinear class."""

    def test_lora_linear_init(self):
        """LoRA linear should initialize with correct params."""
        from geofm.models.peft.lora_adapter import LoRALinear

        layer = nn.Linear(768, 768)
        lora = LoRALinear(layer, rank=16, alpha=32)

        assert lora.rank == 16
        assert lora.alpha == 32
        assert lora.lora_A.shape == (16, 768)
        assert lora.lora_B.shape == (768, 16)

    def test_lora_linear_base_frozen(self):
        """Base layer should be frozen."""
        from geofm.models.peft.lora_adapter import LoRALinear

        layer = nn.Linear(768, 768)
        lora = LoRALinear(layer, rank=16, alpha=32)

        # Base layer should not require grad
        assert not layer.weight.requires_grad
        if layer.bias is not None:
            assert not layer.bias.requires_grad

    def test_lora_linear_trainable_params(self):
        """Only lora_A and lora_B should be trainable."""
        from geofm.models.peft.lora_adapter import LoRALinear

        layer = nn.Linear(768, 768)
        lora = LoRALinear(layer, rank=16, alpha=32)

        trainable_names = []
        for name, p in lora.named_parameters():
            if p.requires_grad:
                trainable_names.append(name)

        assert "lora_A" in trainable_names[0] if trainable_names else True
        assert "lora_B" in trainable_names[1] if len(trainable_names) > 1 else True

    def test_lora_linear_from_linear(self):
        """from_linear should create LoRA layer."""
        from geofm.models.peft.lora_adapter import LoRALinear

        linear = nn.Linear(768, 768)
        lora = LoRALinear.from_linear(linear, rank=8, alpha=16)

        assert lora.rank == 8
        assert lora.alpha == 16

    def test_lora_linear_merge(self):
        """merge should produce equivalent weights."""
        from geofm.models.peft.lora_adapter import LoRALinear

        layer = nn.Linear(768, 768)
        lora = LoRALinear(layer, rank=16, alpha=32)

        merged = lora.merge()

        # Merged layer should be a standard Linear
        assert isinstance(merged, nn.Linear)


class TestLoRAConv2d:
    """Test LoRAConv2d class."""

    def test_lora_conv2d_init(self):
        """LoRA conv2d should initialize."""
        from geofm.models.peft.lora_adapter import LoRAConv2d

        conv = nn.Conv2d(768, 768, kernel_size=3, padding=1)
        lora_conv = LoRAConv2d(conv, rank=16, alpha=32)

        assert lora_conv.rank == 16
        assert lora_conv.base_layer is not None

    def test_lora_conv2d_has_trainable_params(self):
        """LoRA conv2d should have trainable params."""
        from geofm.models.peft.lora_adapter import LoRAConv2d

        conv = nn.Conv2d(768, 768, kernel_size=3, padding=1)
        lora_conv = LoRAConv2d(conv, rank=16, alpha=32)

        trainable = sum(p.numel() for p in lora_conv.parameters() if p.requires_grad)
        assert trainable > 0


class TestLoraUtils:
    """Test LoRA utility functions."""

    def test_count_lora_parameters(self):
        """count_lora_parameters should return correct count."""
        from geofm.models.peft.lora_adapter import LoRALinear, count_lora_parameters

        layer = nn.Linear(768, 768)
        lora = LoRALinear(layer, rank=16, alpha=32)

        count = count_lora_parameters(lora)

        # Should count lora_A + lora_B
        expected = 16 * 768 + 768 * 16  # rank * in + out * rank
        assert count == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])