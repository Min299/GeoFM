"""tests/test_hybrid_path.py

Tier 1: Architecture Integrity Tests

Verify hybrid model path:
Input → LoRA TerraMind → Feature Adapter → Decoder → Output
"""
import pytest
import torch
import torch.nn as nn

from geofm.models.backbones import build_backbone
from geofm.models.peft import TerraMindLoRA, TaskFeatureAdapter, HybridAdapter


class TestHybridPath:
    """Verify hybrid model forward path."""

    @pytest.fixture
    def hybrid_model(self):
        """Create hybrid model."""
        backbone = build_backbone("terramind_base")
        backbone.freeze()

        # Get feature dimension
        feature_dim = getattr(backbone, "feature_dim", 768)
        feature_channels = [feature_dim] * 4

        # Build LoRA
        lora_backbone = TerraMindLoRA(
            backbone,
            rank=8,
            alpha=8,
            freeze_backbone=True,
        )

        # Build feature adapter
        feature_adapter = TaskFeatureAdapter(
            in_channels=feature_channels,
            out_channels=1,
        )

        # Build hybrid
        model = HybridAdapter(
            backbone=lora_backbone,
            feature_adapter=feature_adapter,
        )

        return model

    def test_hybrid_creates_successfully(self, hybrid_model):
        """Hybrid model should create without errors."""
        assert hybrid_model is not None
        assert hasattr(hybrid_model, 'backbone')
        assert hasattr(hybrid_model, 'feature_adapter')

    def test_forward_path_executes(self, hybrid_model):
        """Forward pass should execute without errors."""
        x = torch.randn(1, 12, 224, 224)  # S2L2A format

        # Should not raise
        try:
            output = hybrid_model(x)
        except Exception as e:
            pytest.fail(f"Forward pass failed: {e}")

    def test_all_stages_contribute_to_output(self, hybrid_model):
        """All stages should contribute to output (not just one)."""
        x = torch.randn(1, 12, 224, 224)

        # Get baseline output
        output1 = hybrid_model(x)

        # Save original state
        original_state = {
            k: v.clone()
            for k, v in hybrid_model.state_dict().items()
        }

        # Perturb feature adapter
        for name, param in hybrid_model.feature_adapter.named_parameters():
            param.data += torch.randn_like(param) * 0.1

        output2 = hybrid_model(x)

        # Restore
        hybrid_model.load_state_dict(original_state)

        # Output should change
        diff = (output1 - output2).abs().mean()
        assert diff > 1e-6, "Feature adapter not contributing to output"


class TestHybridPathWithHooks:
    """Verify path using forward hooks."""

    @pytest.fixture
    def hooked_model(self):
        """Create model with hooks to track execution."""
        backbone = build_backbone("terramind_base")
        backbone.freeze()

        feature_dim = getattr(backbone, "feature_dim", 768)
        feature_channels = [feature_dim] * 4

        lora_backbone = TerraMindLoRA(
            backbone,
            rank=8,
            alpha=8,
            freeze_backbone=True,
        )

        feature_adapter = TaskFeatureAdapter(
            in_channels=feature_channels,
            out_channels=1,
        )

        model = HybridAdapter(
            backbone=lora_backbone,
            feature_adapter=feature_adapter,
        )

        return model

    def test_lora_executed_in_forward(self):
        """Verify LoRA layers are called during forward."""
        backbone = build_backbone("terramind_base")
        backbone.freeze()

        lora_backbone = TerraMindLoRA(
            backbone,
            rank=8,
            alpha=8,
            freeze_backbone=True,
        )

        # Track if LoRA forward is called
        call_count = 0

        def hook_fn(module, input, output):
            nonlocal call_count
            call_count += 1

        # Register hooks on all LoRA layers
        handles = []
        for module in lora_backbone.model.modules():
            if isinstance(module, LoRALinear):
                handle = module.register_forward_hook(hook_fn)
                handles.append(handle)

        # Run forward
        x = torch.randn(1, 12, 224, 224)
        _ = lora_backbone(x)

        # Remove hooks
        for handle in handles:
            handle.remove()

        # Should have called LoRA forward
        assert call_count > 0, "LoRA forward not called"


class TestHybridGradientFlow:
    """Verify gradients flow through hybrid path."""

    @pytest.fixture
    def hybrid_model(self):
        """Create hybrid model."""
        backbone = build_backbone("terramind_base")
        backbone.freeze()

        feature_dim = getattr(backbone, "feature_dim", 768)
        feature_channels = [feature_dim] * 4

        lora_backbone = TerraMindLoRA(
            backbone,
            rank=8,
            alpha=8,
            freeze_backbone=True,
        )

        feature_adapter = TaskFeatureAdapter(
            in_channels=feature_channels,
            out_channels=1,
        )

        model = HybridAdapter(
            backbone=lora_backbone,
            feature_adapter=feature_adapter,
        )

        return model

    def test_gradients_exist_in_lora(self, hybrid_model):
        """Gradients should exist in LoRA layers after backward."""
        x = torch.randn(1, 12, 224, 224)
        target = torch.randint(0, 2, (1, 224, 224)).float()

        output = hybrid_model(x)
        loss = nn.functional.binary_cross_entropy_with_logits(output, target.unsqueeze(1))
        loss.backward()

        # Check LoRA gradients
        has_grad = False
        for module in hybrid_model.backbone.modules():
            if isinstance(module, LoRALinear):
                if module.lora_A.grad is not None and module.lora_A.grad.abs().sum() > 0:
                    has_grad = True
                    break

        assert has_grad, "No gradients in LoRA layers"

    def test_gradients_exist_in_feature_adapter(self, hybrid_model):
        """Gradients should exist in feature adapter."""
        x = torch.randn(1, 12, 224, 224)
        target = torch.randint(0, 2, (1, 224, 224)).float()

        output = hybrid_model(x)
        loss = nn.functional.binary_cross_entropy_with_logits(output, target.unsqueeze(1))
        loss.backward()

        # Check feature adapter gradients
        has_grad = False
        for param in hybrid_model.feature_adapter.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "No gradients in feature adapter"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])