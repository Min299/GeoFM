"""tests/test_lora_layer.py

Tests for LoRA layer implementation.
"""
import pytest
import torch
import torch.nn as nn

from geofm.models.peft import LoRALinear, LoRAConfig


class TestLoRALinear:
    """Test LoRA Linear layer."""

    def test_lora_creates_correct_shapes(self):
        """LoRA matrices should have correct dimensions."""
        linear = nn.Linear(768, 2304)  # QKV layer
        lora = LoRALinear(linear, rank=16, alpha=16)

        assert lora.lora_A.shape == (16, 768)  # (rank, in_features)
        assert lora.lora_B.shape == (2304, 16)  # (out_features, rank)
        assert lora.scale == 1.0  # alpha / rank = 16 / 16

    def test_lora_initialization(self):
        """LoRA A should be initialized with kaiming, B should be zeros."""
        linear = nn.Linear(768, 768)
        lora = LoRALinear(linear, rank=16, alpha=16)

        # A should not be all zeros (kaiming init)
        assert lora.lora_A.abs().sum() > 0

        # B should be all zeros
        assert lora.lora_B.abs().sum() == 0

    def test_original_weights_frozen(self):
        """Original linear weights should be frozen."""
        linear = nn.Linear(768, 768)
        lora = LoRALinear(linear, rank=16, alpha=16)

        assert not lora.linear.weight.requires_grad
        if lora.linear.bias is not None:
            assert not lora.linear.bias.requires_grad

    def test_lora_params_trainable(self):
        """LoRA A and B should be trainable."""
        linear = nn.Linear(768, 768)
        lora = LoRALinear(linear, rank=16, alpha=16)

        assert lora.lora_A.requires_grad
        assert lora.lora_B.requires_grad

    def test_forward_equals_base_plus_delta(self):
        """Forward should compute base + scale * B @ A @ x."""
        linear = nn.Linear(768, 768)
        # Set identity weights for easy verification
        linear.weight.data = torch.eye(768)
        linear.bias.data = torch.zeros(768)

        lora = LoRALinear(linear, rank=16, alpha=16)

        x = torch.randn(2, 768)

        # Set LoRA to identity (A=I, B=I, scale=1)
        with torch.no_grad():
            lora.lora_A.copy_(torch.eye(16, 768))
            lora.lora_B.copy_(torch.eye(768, 16))

        output = lora(x)

        # Base output: x (identity weights)
        base = linear(x)

        # LoRA delta: scale * B @ A @ x = 1 * I @ I @ x = x
        delta = lora.scale * (x @ lora.lora_A.t()) @ lora.lora_B.t()

        expected = base + delta

        assert torch.allclose(output, expected, atol=1e-5)

    def test_different_rank_alpha(self):
        """Should work with different rank and alpha values."""
        linear = nn.Linear(768, 768)

        # Test rank=8, alpha=32
        lora = LoRALinear(linear, rank=8, alpha=32)
        assert lora.rank == 8
        assert lora.alpha == 32
        assert lora.scale == 4.0  # 32 / 8

        # Test rank=64, alpha=64
        lora = LoRALinear(linear, rank=64, alpha=64)
        assert lora.rank == 64
        assert lora.alpha == 64
        assert lora.scale == 1.0  # 64 / 64


class TestLoRAConfig:
    """Test LoRA configuration."""

    def test_default_config(self):
        """Default config should have sensible values."""
        config = LoRAConfig()

        assert config.rank == 16
        assert config.alpha == 16
        assert config.target_modules == ["qkv", "proj"]
        assert config.dropout == 0.0

    def test_custom_config(self):
        """Custom config should override defaults."""
        config = LoRAConfig(rank=32, alpha=64, target_modules=["qkv"])

        assert config.rank == 32
        assert config.alpha == 64
        assert config.target_modules == ["qkv"]


class TestLoRAGradientFlow:
    """Test gradient flow through LoRA layers."""

    def test_gradients_flow_through_lora(self):
        """Gradients should flow through LoRA parameters."""
        linear = nn.Linear(768, 768)
        lora = LoRALinear(linear, rank=16, alpha=16)

        x = torch.randn(2, 768)
        output = lora(x)
        loss = output.mean()
        loss.backward()

        assert lora.lora_A.grad is not None
        assert lora.lora_B.grad is not None
        # Gradients should be non-zero
        assert lora.lora_A.grad.abs().sum() > 0 or lora.lora_B.grad.abs().sum() > 0

    def test_no_gradients_on_base(self):
        """Base weights should not receive gradients."""
        linear = nn.Linear(768, 768)
        lora = LoRALinear(linear, rank=16, alpha=16)

        x = torch.randn(2, 768)
        output = lora(x)
        loss = output.mean()
        loss.backward()

        assert lora.linear.weight.grad is None


class TestLoRAMerge:
    """Test LoRA weight merging."""

    def test_merge_updates_weights(self):
        """Merging should update the base weight."""
        linear = nn.Linear(768, 768)
        lora = LoRALinear(linear, rank=16, alpha=16)

        original_weight = lora.linear.weight.data.clone()

        # Set LoRA to add a known delta
        with torch.no_grad():
            lora.lora_A.copy_(torch.ones(16, 768) * 0.01)
            lora.lora_B.copy_(torch.ones(768, 16) * 0.01)

        lora.merge()

        # Weight should have changed
        assert not torch.allclose(lora.linear.weight.data, original_weight)

    def test_merge_makes_lora_frozen(self):
        """After merge, LoRA params should be frozen."""
        linear = nn.Linear(768, 768)
        lora = LoRALinear(linear, rank=16, alpha=16)

        lora.merge()

        assert not lora.lora_A.requires_grad
        assert not lora.lora_B.requires_grad


if __name__ == "__main__":
    pytest.main([__file__, "-v"])