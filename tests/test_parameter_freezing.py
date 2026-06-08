"""tests/test_parameter_freezing.py

Tier 1: Architecture Integrity Tests

Verify:
- Backbone parameters are frozen
- Only LoRA parameters are trainable

This catches the most common PEFT mistake.
"""
import pytest
import torch
import torch.nn as nn

from geofm.models.backbones import build_backbone
from geofm.models.peft import TerraMindLoRA, LoRALinear, freeze_all_except_lora


class TestParameterFreezing:
    """Verify parameter freezing is correct."""

    @pytest.fixture
    def lora_model(self):
        """Create LoRA model with frozen backbone."""
        backbone = build_backbone("terramind_base")
        return TerraMindLoRA(backbone, rank=16, alpha=16, freeze_backbone=True)

    def test_backbone_frozen(self, lora_model):
        """Verify all backbone parameters are frozen."""
        model = lora_model.model

        trainable = []
        frozen = []

        for name, param in model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                trainable.append(name)
            else:
                if param.requires_grad:
                    frozen.append(name)

        # All backbone params should be frozen
        assert len(frozen) == 0, f"Found trainable backbone params: {frozen[:5]}"

    def test_lora_trainable(self, lora_model):
        """Verify LoRA parameters are trainable."""
        trainable_params = sum(
            p.numel() for p in lora_model.model.parameters()
            if p.requires_grad
        )

        lora_params = lora_model.count_lora_params()

        assert trainable_params > 0, "No trainable parameters"
        assert lora_params > 0, "No LoRA parameters"
        assert trainable_params == lora_params, \
            f"Trainable ({trainable_params}) != LoRA params ({lora_params})"

    def test_original_weights_frozen(self, lora_model):
        """Verify original weight matrices are frozen."""
        for module in lora_model.model.modules():
            if isinstance(module, LoRALinear):
                # Original weight should be frozen
                assert not module.linear.weight.requires_grad, \
                    "Original weight should be frozen"

                # LoRA should be trainable
                assert module.lora_A.requires_grad, "lora_A should be trainable"
                assert module.lora_B.requires_grad, "lora_B should be trainable"

    def test_trainable_ratio_under_threshold(self, lora_model):
        """Verify trainable ratio is under 5%."""
        total = sum(p.numel() for p in lora_model.model.parameters())
        trainable = sum(p.numel() for p in lora_model.model.parameters() if p.requires_grad)

        ratio = trainable / total if total > 0 else 0

        assert ratio < 0.05, f"Trainable ratio {ratio:.2%} exceeds 5% threshold"
        print(f"Trainable ratio: {ratio:.2%} ({trainable:,} / {total:,})")

    def test_verify_lora_only_trainable(self, lora_model):
        """Test the verify_lora_only_trainable function."""
        from geofm.models.peft import verify_lora_only_trainable

        result = verify_lora_only_trainable(lora_model.model)
        assert result is True, "verify_lora_only_trainable returned False"

    def test_freeze_all_except_lora_function(self):
        """Test the freeze_all_except_lora helper function."""
        backbone = build_backbone("terramind_base")

        # Manually inject LoRA
        from geofm.models.peft import LoRALinear

        inner = backbone._model if hasattr(backbone, '_model') else backbone
        for block in inner.encoder:
            if hasattr(block, 'attn'):
                if hasattr(block.attn, 'qkv'):
                    block.attn.qkv = LoRALinear(block.attn.qkv, rank=8, alpha=8)

        # Apply freeze
        freeze_all_except_lora(backbone)

        # Verify
        trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        assert trainable > 0, "No trainable params after freeze_all_except_lora"


class TestFreezingAfterMerge:
    """Test freezing behavior after merging LoRA."""

    @pytest.fixture
    def lora_model_for_merge(self):
        """Create LoRA model for merge test."""
        backbone = build_backbone("terramind_base")
        return TerraMindLoRA(backbone, rank=16, alpha=16, freeze_backbone=True)

    def test_merged_weights_trainable(self, lora_model_for_merge):
        """After merge, weights become trainable."""
        lora_model_for_merge.merge_weights()

        for module in lora_model_for_merge.model.modules():
            if isinstance(module, LoRALinear):
                # After merge, the original weight becomes trainable
                # (This is by design - merge makes LoRA permanent)
                assert module.linear.weight.requires_grad, \
                    "Merged weight should be trainable"
                # And LoRA params become frozen
                assert not module.lora_A.requires_grad, "lora_A should be frozen after merge"
                assert not module.lora_B.requires_grad, "lora_B should be frozen after merge"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])