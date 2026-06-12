"""tests/test_real_lora_pipeline.py

Tests for real LoRA pipeline.
"""
import pytest
import torch
import torch.nn as nn


class TestRealLoRAPipeline:
    """Test LoRA forward pipeline."""

    def test_lora_linear_forward(self):
        """LoRA linear should work with 3D input."""
        from geofm.models.peft.lora_adapter import LoRALinear

        layer = nn.Linear(768, 768)
        lora = LoRALinear(layer, rank=16, alpha=32)

        x = torch.randn(2, 10, 768)

        y = lora(x)

        assert y.shape == x.shape

    def test_lora_linear_2d_input(self):
        """LoRA linear should work with 2D input."""
        from geofm.models.peft.lora_adapter import LoRALinear

        layer = nn.Linear(768, 768)
        lora = LoRALinear(layer, rank=16, alpha=32)

        x = torch.randn(2, 768)

        y = lora(x)

        assert y.shape == x.shape

    def test_lora_injector_single_block(self):
        """LoRA injector should work on single block."""
        from geofm.models.peft.lora_injector import LoRAInjector

        class DummyAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.qkv = nn.Linear(768, 2304)
                self.proj = nn.Linear(768, 768)

        class DummyBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = DummyAttn()

        model = DummyBlock()
        injector = LoRAInjector()

        replaced = injector.inject(model)

        assert len(replaced) == 2

    def test_lora_forward_after_injection(self):
        """Model should forward after LoRA injection."""
        from geofm.models.peft.lora_injector import LoRAInjector

        class DummyAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.qkv = nn.Linear(768, 2304)
                self.proj = nn.Linear(768, 768)

            def forward(self, x):
                # QKV projection
                qkv = self.qkv(x)
                # Simplified attention
                return self.proj(x)

        class DummyBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = DummyAttn()

            def forward(self, x):
                return self.attn(x)

        model = DummyBlock()
        injector = LoRAInjector()
        injector.inject(model)

        x = torch.randn(2, 10, 768)
        y = model(x)

        assert y.shape == x.shape

    def test_lora_frozen_after_injection(self):
        """Original params should be frozen after injection."""
        from geofm.models.peft.lora_injector import LoRAInjector

        class DummyAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.qkv = nn.Linear(768, 768)
                self.proj = nn.Linear(768, 768)

        class DummyBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = DummyAttn()

        model = DummyBlock()
        injector = LoRAInjector()
        injector.inject(model)

        # Original layer should be frozen
        assert not model.attn.qkv.base_layer.weight.requires_grad

    def test_lora_trainable_params(self):
        """LoRA params should be trainable."""
        from geofm.models.peft.lora_injector import LoRAInjector

        class DummyAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.qkv = nn.Linear(768, 768)
                self.proj = nn.Linear(768, 768)

        class DummyBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = DummyAttn()

        model = DummyBlock()
        injector = LoRAInjector()
        injector.inject(model)
        injector.freeze_except_lora(model)

        # Count trainable params
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Should have LoRA params
        assert trainable > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])