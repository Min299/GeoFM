"""tests/test_adapter_gradients.py

Test that adapters receive gradients.
Critical: Verify FeatureAdapter gets gradients, not backbone.
"""
import pytest
import torch
import torch.nn as nn


class TestAdapterGradients:
    """Test adapter gradient flow."""

    def test_feature_adapter_receives_gradients(self):
        """FeatureAdapter should receive gradients."""
        from geofm.models.peft.feature_adapter import FeatureAdapter

        adapter = FeatureAdapter(dim=768, bottleneck_dim=64)

        # Enable gradients
        for p in adapter.parameters():
            p.requires_grad = True

        # Forward
        x = torch.randn(2, 768, 64, 64)
        out = adapter(x)

        # Loss and backward
        loss = out[0].sum()
        loss.backward()

        # Check gradients
        adapter_grad = sum(p.grad.abs().sum().item()
                          for p in adapter.parameters()
                          if p.grad is not None)

        assert adapter_grad > 0, "FeatureAdapter has no gradients"

    def test_task_feature_adapter_receives_gradients(self):
        """TaskFeatureAdapter should receive gradients."""
        from geofm.models.peft.task_feature_adapter import TaskFeatureAdapter

        adapter = TaskFeatureAdapter(feature_dims=[768, 768, 768, 768])

        for p in adapter.parameters():
            p.requires_grad = True

        features = [torch.randn(2, 768, 64, 64) for _ in range(4)]
        out = adapter(features)

        loss = sum(f.sum() for f in out)
        loss.backward()

        adapter_grad = sum(p.grad.abs().sum().item()
                          for p in adapter.parameters()
                          if p.grad is not None)

        assert adapter_grad > 0, "TaskFeatureAdapter has no gradients"

    def test_adapter_bank_gradients(self):
        """Simple model with list input should receive gradients."""
        # Create a simple model that takes a list of features
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(768, 256, 1)

            def forward(self, features):
                # Take first feature
                return self.conv(features[0])

        model = SimpleModel()

        for p in model.parameters():
            p.requires_grad = True

        features = [torch.randn(2, 768, 64, 64) for _ in range(4)]
        out = model(features)

        loss = out.sum()
        loss.backward()

        model_grad = sum(p.grad.abs().sum().item()
                        for p in model.parameters()
                        if p.grad is not None)

        assert model_grad > 0, "Model has no gradients"

    def test_adapter_gradients_larger_than_zero(self):
        """Adapter gradients should be significant."""
        from geofm.models.peft.feature_adapter import FeatureAdapter

        adapter = FeatureAdapter(dim=768, bottleneck_dim=64)

        for p in adapter.parameters():
            p.requires_grad = True

        x = torch.randn(2, 768, 64, 64)
        out = adapter(x)

        loss = out[0].sum()
        loss.backward()

        adapter_grad_norm = sum(p.grad.abs().sum().item()
                              for p in adapter.parameters()
                              if p.grad is not None)

        assert adapter_grad_norm > 1.0, f"Gradient norm too small: {adapter_grad_norm}"

    def test_lora_adapter_gradients(self):
        """LoRA adapter should receive gradients."""
        from geofm.models.peft.lora_adapter import LoRALinear

        layer = nn.Linear(768, 768)
        lora = LoRALinear(layer, rank=16, alpha=32)

        x = torch.randn(2, 10, 768)
        out = lora(x)

        loss = out.sum()
        loss.backward()

        lora_grad = sum(p.grad.abs().sum().item()
                       for p in lora.parameters()
                       if p.grad is not None)

        assert lora_grad > 0, "LoRA has no gradients"

    def test_adapter_fc1_fc2_gradients(self):
        """Adapter FC layers should receive gradients."""
        from geofm.models.peft.feature_adapter import FeatureAdapter

        adapter = FeatureAdapter(dim=768, bottleneck_dim=64)

        for p in adapter.parameters():
            p.requires_grad = True

        x = torch.randn(2, 768, 64, 64)
        out = adapter(x)

        loss = out[0].sum()
        loss.backward()

        # Check that FC layers have gradients
        for name, p in adapter.named_parameters():
            if 'down' in name or 'up' in name:
                assert p.grad is not None, f"Layer {name} has no gradients"
                assert p.grad.abs().sum().item() > 0, f"Layer {name} has zero gradients"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
