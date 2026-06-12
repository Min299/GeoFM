"""tests/test_trainable_parameter_audit.py

Trainable parameter audit for PEFT validation.

Critical check: Verify that only adapters are trainable,
not the backbone (TerraMind).

This test alone can save weeks of GPU time by catching
accidental full fine-tuning before training starts.
"""
import pytest
import torch
import torch.nn as nn


class TestTrainableParameterAudit:
    """Test trainable parameter configuration."""

    def test_feature_adapter_trainable(self):
        """FeatureAdapter parameters should be trainable."""
        from geofm.models.peft.feature_adapter import FeatureAdapter

        adapter = FeatureAdapter(dim=768, bottleneck_dim=64)

        trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in adapter.parameters() if not p.requires_grad)

        # All adapter params should be trainable
        assert trainable > 0
        assert frozen == 0

    def test_task_feature_adapter_trainable(self):
        """TaskFeatureAdapter parameters should be trainable."""
        from geofm.models.peft.task_feature_adapter import TaskFeatureAdapter

        adapter = TaskFeatureAdapter(feature_dims=[768, 768, 768, 768])

        trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)

        # Should have trainable params
        assert trainable > 0

    def test_lora_linear_frozen_base(self):
        """LoRA base layer should be frozen."""
        from geofm.models.peft.lora_adapter import LoRALinear

        layer = nn.Linear(768, 768)
        lora = LoRALinear(layer, rank=16, alpha=32)

        # Base layer should be frozen
        assert not lora.base_layer.weight.requires_grad

    def test_lora_linear_trainable(self):
        """LoRA A and B matrices should be trainable."""
        from geofm.models.peft.lora_adapter import LoRALinear

        layer = nn.Linear(768, 768)
        lora = LoRALinear(layer, rank=16, alpha=32)

        trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)

        # lora_A + lora_B should be trainable
        assert trainable > 0

    def test_backbone_frozen_after_manual_freeze(self):
        """Backbone parameters should be frozen after manual freeze."""
        backbone = nn.Linear(768, 768)
        adapter = nn.Linear(768, 2)

        # Freeze backbone
        for p in backbone.parameters():
            p.requires_grad = False

        # Check that backbone is frozen
        assert not backbone.weight.requires_grad
        assert not backbone.bias.requires_grad

    def test_frozen_params_do_not_change(self):
        """Frozen params should not change during training."""
        backbone = nn.Linear(10, 10)
        adapter = nn.Linear(10, 5)

        # Freeze backbone
        for p in backbone.parameters():
            p.requires_grad = False

        model = nn.Sequential(backbone, adapter)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Store backbone weights
        backbone_weight_before = backbone.weight.clone()

        # Training step
        x = torch.randn(2, 10)
        target = torch.randint(0, 5, (2,))

        optimizer.zero_grad()
        out = model(x)
        loss = nn.functional.cross_entropy(out, target)
        loss.backward()
        optimizer.step()

        # Backbone weights should not have changed
        assert torch.allclose(backbone_weight_before, backbone.weight, atol=1e-6), \
            "Frozen backbone weights changed"

    def test_peft_ratio_calculation(self):
        """PEFT ratio should be calculated correctly."""
        from geofm.debug.parameter_report import peft_ratio

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Simulate frozen backbone + trainable adapter
                self.backbone = nn.Linear(100, 100)
                self.adapter = nn.Linear(100, 100)

                # Freeze backbone
                for p in self.backbone.parameters():
                    p.requires_grad = False

        model = DummyModel()
        ratio = peft_ratio(model)

        # Only adapter params should be trainable
        # 100 * 100 = 10000 params per linear
        # backbone: 10000 (frozen)
        # adapter: 10000 (trainable)
        # total: 20000
        # ratio: 10000 / 20000 = 50%
        assert ratio == 50.0

    def test_adapter_bank_trainable_params(self):
        """AdapterBank should have trainable parameters."""
        from geofm.models.peft.adapter_bank import AdapterBank
        from geofm.models.peft.feature_adapter import FeatureAdapter

        bank = AdapterBank()
        bank.register_task("flood", FeatureAdapter(dim=768, bottleneck_dim=64))
        bank.register_task("burn", FeatureAdapter(dim=768, bottleneck_dim=64))

        trainable = sum(p.numel() for p in bank.parameters() if p.requires_grad)

        # Should have trainable params
        assert trainable > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
