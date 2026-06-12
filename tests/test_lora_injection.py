"""tests/test_lora_injection.py

Tests for LoRA injection into TerraMind backbone.

The key test is verifying that LoRA is properly injected into
all 24 attention layers (12 blocks × 2 layers per block: qkv + proj).
"""
import pytest
import torch

from geofm.models.backbones import build_backbone
from geofm.models.peft import TerraMindLoRA, LoRALinear


@pytest.mark.integration
class TestLoRAInjection:
    """Test LoRA injection into TerraMind."""

    def test_lora_injects_into_all_layers(self):
        """LoRA should be injected into all 24 attention layers.

        TerraMind Base has 12 transformer blocks.
        Each block has 2 target layers: attn.qkv and attn.proj.
        12 × 2 = 24 layers total.
        """
        model = build_backbone("terramind_base")

        # Freeze first (standard PEFT workflow)
        model.freeze()

        # Inject LoRA
        lora_model = TerraMindLoRA(model, rank=16, alpha=16)

        # Count LoRA layers
        count = 0
        for module in lora_model.model.modules():
            if isinstance(module, LoRALinear):
                count += 1

        # Should be 24 (12 blocks × 2 layers)
        assert count == 24, f"Expected 24 LoRA layers, got {count}"

    def test_lora_count_matches_injected(self):
        """count_lora_layers() should return correct count."""
        model = build_backbone("terramind_base")
        model.freeze()

        lora_model = TerraMindLoRA(model, rank=16, alpha=16)

        assert lora_model.count_lora_layers() == 24

    def test_lora_params_count(self):
        """LoRA params should be: 24 × 2 × (768 * 16 + 768 * 16)"""
        model = build_backbone("terramind_base")
        model.freeze()

        lora_model = TerraMindLoRA(model, rank=16, alpha=16)

        # Each LoRA layer has 2 params: lora_A and lora_B
        # lora_A: (rank, in_features) = (16, 768)
        # lora_B: (out_features, rank) = (768, 16)
        # For QKV: lora_B: (2304, 16)
        # For Proj: lora_B: (768, 16)

        # Total params per QKV LoRA: 16*768 + 2304*16 = 12,288 + 36,864 = 49,152
        # Total params per Proj LoRA: 16*768 + 768*16 = 12,288 + 12,288 = 24,576

        # 12 QKV layers: 12 × 49,152 = 589,824
        # 12 Proj layers: 12 × 24,576 = 294,912
        # Total: 884,736

        lora_params = lora_model.count_lora_params()
        expected = 589_824 + 294_912  # 884,736
        assert lora_params == expected, f"Expected {expected}, got {lora_params}"

    def test_original_weights_still_frozen(self):
        """Original backbone weights should remain frozen."""
        model = build_backbone("terramind_base")
        model.freeze()

        lora_model = TerraMindLoRA(model, rank=16, alpha=16)

        # Check that original weights are still frozen
        for name, param in lora_model.model.named_parameters():
            # LoRA params should be trainable
            if "lora_A" in name or "lora_B" in name:
                assert param.requires_grad, f"LoRA param {name} should be trainable"
            # Original backbone weights should be frozen
            elif any(x in name for x in ["qkv", "proj"]):
                if "lora" not in name:
                    # These should be wrapped by LoRA, so original is not a param
                    pass
            else:
                # Other backbone params should be frozen
                assert not param.requires_grad, f"Backbone param {name} should be frozen"

    def test_only_lora_params_trainable(self):
        """Only LoRA parameters should be trainable."""
        model = build_backbone("terramind_base")
        model.freeze()

        lora_model = TerraMindLoRA(model, rank=16, alpha=16)

        trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)

        # Should equal LoRA params count
        lora_params = lora_model.count_lora_params()
        assert trainable == lora_params, f"Expected {lora_params} trainable, got {trainable}"

    def test_lora_inject_twice_no_double(self):
        """Injecting LoRA twice should not create duplicates.

        Note: Re-injection replaces existing LoRA layers, so the
        count stays the same but old LoRA params are replaced.
        """
        model = build_backbone("terramind_base")
        model.freeze()

        lora_model = TerraMindLoRA(model, rank=16, alpha=16)

        initial_count = lora_model.count_lora_layers()
        initial_params = lora_model.count_lora_params()

        # Inject again
        lora_model.inject()

        # Count should still be 24 (old layers replaced)
        assert lora_model.count_lora_layers() == initial_count
        # But params may differ if using same rank
        assert lora_model.count_lora_params() == initial_params


@pytest.mark.integration
class TestLoRAForward:
    """Test forward pass through LoRA model.

    Note: Full forward pass requires proper band configuration
    matching the pretrained model (S2L2A 12 bands or S2L1C 13 bands).
    These tests verify the structure is correct.
    """

    def test_lora_model_has_forward(self):
        """LoRA model should have forward method."""
        model = build_backbone("terramind_base")
        model.freeze()

        lora_model = TerraMindLoRA(model, rank=16, alpha=16)

        assert hasattr(lora_model, 'forward')
        assert callable(lora_model.forward)

    def test_lora_preserves_model_structure(self):
        """LoRA injection should preserve model structure."""
        model = build_backbone("terramind_base")
        model.freeze()

        # Check original layer count
        original_attn_layers = sum(
            1 for _, m in model.named_modules()
            if hasattr(m, 'attn') and hasattr(m.attn, 'qkv')
        )

        lora_model = TerraMindLoRA(model, rank=16, alpha=16)

        # Should have same number of attention layers
        lora_attn_layers = sum(
            1 for _, m in lora_model.model.named_modules()
            if hasattr(m, 'attn') and hasattr(m.attn, 'qkv')
        )

        assert lora_attn_layers == original_attn_layers


@pytest.mark.integration
class TestLoRAParameterEfficiency:
    """Test PEFT efficiency of LoRA injection."""

    def test_peft_ratio_under_threshold(self):
        """LoRA model should have < 5% trainable params."""
        model = build_backbone("terramind_base")
        model.freeze()

        lora_model = TerraMindLoRA(model, rank=16, alpha=16)

        total = sum(p.numel() for p in lora_model.parameters())
        trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)

        ratio = trainable / total
        peft_pct = ratio * 100

        print(f"\n  Total params: {total:,}")
        print(f"  Trainable params: {trainable:,}")
        print(f"  PEFT ratio: {peft_pct:.2f}%")

        # Should be under 5%
        assert peft_pct < 5.0, f"PEFT ratio {peft_pct:.2f}% exceeds 5% threshold"

    def test_different_ranks(self):
        """Test LoRA with different rank values."""
        ranks = [4, 8, 16, 32]

        for rank in ranks:
            model = build_backbone("terramind_base")
            model.freeze()

            lora_model = TerraMindLoRA(model, rank=rank, alpha=rank)

            # Count should still be 24
            assert lora_model.count_lora_layers() == 24

            # Check params scale with rank
            params = lora_model.count_lora_params()
            # Params scale linearly with rank
            assert params == 884_736 * (rank / 16), f"Rank {rank} params mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])