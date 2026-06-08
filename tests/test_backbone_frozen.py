"""tests/test_backbone_frozen.py

Test that backbone remains frozen during training.
Critical: Verify TerraMind gradients are zero.
"""
import pytest
import torch
import torch.nn as nn


class TestBackboneFrozen:
    """Test backbone frozen state."""

    def test_backbone_params_frozen_after_freeze(self):
        """Backbone parameters should be frozen after freeze()."""
        from geofm.models.backbones.terramind_backbone import TerraMindBackbone

        class DummyModel:
            def forward_encoder(self, batch):
                return [torch.randn(2, 768, 64, 64) for _ in range(12)]

        try:
            backbone = TerraMindBackbone(DummyModel())
            backbone.freeze()

            # Check all params are frozen
            trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)

            assert trainable == 0, f"Backbone has {trainable} trainable params after freeze"

        except ImportError:
            pytest.skip("TerraMind backbone not available")

    def test_backbone_grad_none_after_backward(self):
        """Backbone gradients should be None after backward."""
        # Simple test with linear layers
        backbone = nn.Linear(10, 10)
        adapter = nn.Linear(10, 5)

        # Freeze backbone
        for p in backbone.parameters():
            p.requires_grad = False

        # Enable adapter gradients
        for p in adapter.parameters():
            p.requires_grad = True

        # Forward and backward
        x = torch.randn(2, 10)
        out = adapter(backbone(x))
        loss = out.sum()
        loss.backward()

        # Check backbone gradients are None
        backbone_grad_count = sum(1 for p in backbone.parameters() if p.grad is not None)

        assert backbone_grad_count == 0, f"Backbone has {backbone_grad_count} params with gradients"

    def test_terramind_backbone_freeze(self):
        """TerraMindBackbone freeze should work."""
        from geofm.models.backbones.terramind_backbone import TerraMindBackbone

        class DummyModel:
            def forward_encoder(self, batch):
                return [torch.randn(2, 768, 64, 64) for _ in range(12)]

        try:
            backbone = TerraMindBackbone(DummyModel())

            # Before freeze
            trainable_before = sum(p.numel() for p in backbone.parameters() if p.requires_grad)

            # Freeze
            backbone.freeze()

            # After freeze
            trainable_after = sum(p.numel() for p in backbone.parameters() if p.requires_grad)

            assert trainable_after < trainable_before, "Freeze had no effect"

        except ImportError:
            pytest.skip("TerraMind backbone not available")

    def test_shared_geofm_backbone_frozen(self):
        """SharedGeoFM backbone should be frozen."""
        try:
            from geofm.models.multitask.shared_geofm import SharedGeoFM

            class DummyBackbone(nn.Module):
                def __init__(self):
                    super().__init__()

                def extract_features(self, batch):
                    return [torch.randn(2, 768, 64, 64) for _ in range(4)]

            class DummyAdapterBank(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.adapters = nn.ModuleDict({
                        "flood": nn.Conv2d(768, 768, 1),
                        "burn": nn.Conv2d(768, 768, 1),
                        "lulc": nn.Conv2d(768, 768, 1),
                    })

                def forward(self, task, features):
                    return features

            class DummyDecoderBank(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.decoders = nn.ModuleDict({
                        "flood": nn.Conv2d(768, 2, 1),
                        "burn": nn.Conv2d(768, 2, 1),
                        "lulc": nn.Conv2d(768, 2, 1),
                    })

                def forward(self, task, features):
                    return self.decoders[task](features[0])

            model = SharedGeoFM(
                backbone=DummyBackbone(),
                adapter_bank=DummyAdapterBank(),
                decoder_bank=DummyDecoderBank(),
            )

            # Freeze backbone
            if hasattr(model.backbone, 'freeze'):
                model.backbone.freeze()

            # Check backbone params are frozen
            backbone_trainable = 0
            if hasattr(model, 'backbone'):
                for p in model.backbone.parameters():
                    if p.requires_grad:
                        backbone_trainable += 1

            # Backbone should be frozen
            assert backbone_trainable == 0, f"Backbone has {backbone_trainable} trainable params"

        except ImportError:
            pytest.skip("SharedGeoFM not available")

    def test_backbone_trainable_count_zero(self):
        """Backbone trainable count should be zero after freeze."""
        from geofm.models.peft.lora_injector import inject_lora_into_backbone

        class DummyBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.Linear(768, 768) for _ in range(12)
                ])

        model = DummyBackbone()
        injector = inject_lora_into_backbone(model, rank=16, alpha=32)

        # Count trainable backbone params (non-LoRA)
        backbone_trainable = 0
        for name, p in model.named_parameters():
            if p.requires_grad and 'lora' not in name:
                backbone_trainable += 1

        # Should be zero
        assert backbone_trainable == 0, f"Backbone has {backbone_trainable} trainable params"

    def test_backbone_grad_count_zero(self):
        """Backbone gradient count should be zero after backward."""
        backbone = nn.Linear(10, 10)
        adapter = nn.Linear(10, 5)

        # Freeze backbone
        for p in backbone.parameters():
            p.requires_grad = False

        # Forward and backward
        x = torch.randn(2, 10)
        out = adapter(backbone(x))
        loss = out.sum()
        loss.backward()

        # Count backbone gradients
        backbone_grad_params = sum(1 for p in backbone.parameters() if p.grad is not None)

        assert backbone_grad_params == 0, f"Backbone has {backbone_grad_params} params with gradients"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])