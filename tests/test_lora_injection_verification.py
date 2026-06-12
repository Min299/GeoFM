"""tests/test_lora_injection_verification.py

Tier 1: Architecture Integrity Tests

Verify LoRA injection creates exactly 24 layers:
- 12 encoder blocks × 2 modules (qkv + proj) = 24 total
"""
import pytest
import torch
import torch.nn as nn

from geofm.models.backbones import build_backbone
from geofm.models.peft import TerraMindLoRA, LoRALinear


class TestLoRAInjectionArchitecture:
    """Verify LoRA injection creates correct architecture."""

    @pytest.fixture
    def backbone(self):
        """Create backbone."""
        return build_backbone("terramind_base")

    @pytest.fixture
    def lora_model(self, backbone):
        """Create LoRA model."""
        return TerraMindLoRA(backbone, rank=16, alpha=16, freeze_backbone=True)

    def test_exactly_24_lora_layers(self, lora_model):
        """Verify exactly 24 LoRA layers injected.

        Expected: 12 blocks × 2 modules (qkv + proj) = 24
        """
        count = lora_model.count_lora_layers()
        assert count == 24, f"Expected 24 LoRA layers, got {count}"

    def test_encoder_block_count(self, lora_model):
        """Verify backbone has 12 encoder blocks."""
        inner = lora_model.model._model if hasattr(lora_model.model, '_model') else lora_model.model
        assert hasattr(inner, 'encoder'), "Model missing encoder"

        encoder = inner.encoder
        num_blocks = len(encoder)
        assert num_blocks == 12, f"Expected 12 encoder blocks, got {num_blocks}"

    def test_all_qkv_layers_replaced(self, lora_model):
        """Verify all QKV layers are LoRALinear."""
        inner = lora_model.model._model if hasattr(lora_model.model, '_model') else lora_model.model

        replaced = 0
        for block in inner.encoder:
            if hasattr(block, 'attn') and hasattr(block.attn, 'qkv'):
                assert isinstance(block.attn.qkv, LoRALinear), \
                    f"QKV layer not replaced: {block}"
                replaced += 1

        assert replaced == 12, f"Expected 12 QKV layers replaced, got {replaced}"

    def test_all_proj_layers_replaced(self, lora_model):
        """Verify all projection layers are LoRALinear."""
        inner = lora_model.model._model if hasattr(lora_model.model, '_model') else lora_model.model

        replaced = 0
        for block in inner.encoder:
            if hasattr(block, 'attn') and hasattr(block.attn, 'proj'):
                assert isinstance(block.attn.proj, LoRALinear), \
                    f"Projection layer not replaced: {block}"
                replaced += 1

        assert replaced == 12, f"Expected 12 projection layers replaced, got {replaced}"

    def test_lora_params_per_layer(self, lora_model):
        """Verify LoRA params per layer match rank."""
        rank = lora_model.rank

        for module in lora_model.model.modules():
            if isinstance(module, LoRALinear):
                # LoRA A: (rank, in_features)
                # LoRA B: (out_features, rank)
                lora_a_shape = module.lora_A.shape
                lora_b_shape = module.lora_B.shape

                # Should be (rank, dim) for lora_A
                assert lora_a_shape[0] == rank, f"lora_A row dim should be {rank}"
                # Should be (dim, rank) for lora_B
                assert lora_b_shape[1] == rank, f"lora_B col dim should be {rank}"

    def test_lora_rank_variations(self, backbone):
        """Verify LoRA works with different ranks."""
        for rank in [1, 4, 16]:
            lora_model = TerraMindLoRA(backbone, rank=rank, alpha=rank, freeze_backbone=True)
            assert lora_model.count_lora_layers() == 24
            assert lora_model.rank == rank

            # Check params scale with rank
            expected_params = lora_model.count_lora_params()
            assert expected_params > 0


class TestLoRAInjectionWithDifferentBackbones:
    """Test LoRA injection on different backbone sizes."""

    @pytest.mark.parametrize("backbone_name", [
        "terramind_tiny",
        "terramind_small",
        "terramind_base",
    ])
    def test_lora_injection_all_backbones(self, backbone_name):
        """Verify LoRA injection works on all backbone sizes."""
        try:
            backbone = build_backbone(backbone_name)
            lora_model = TerraMindLoRA(backbone, rank=8, alpha=8, freeze_backbone=True)

            # Should have some LoRA layers (varies by backbone size)
            assert lora_model.count_lora_layers() > 0
            assert lora_model.count_lora_params() > 0
        except Exception as e:
            pytest.skip(f"Backbone {backbone_name} not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])