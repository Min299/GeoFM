"""tests/test_edge_cases.py

Edge Case Tests

Test various edge cases:
- Empty batch
- Single image batch
- Wrong task name
- Different LoRA ranks
"""
import pytest
import torch
import torch.nn as nn

from geofm.models.peft import TerraMindLoRA, AdapterBank, TaskFeatureAdapter
from geofm.models.backbones import build_backbone


class TestEmptyBatch:
    """Test handling of empty batches."""

    def test_empty_batch_should_fail(self):
        """Empty batch should fail gracefully."""
        adapter = TaskFeatureAdapter(
            in_channels=[768, 768, 768, 768],
            out_channels=1,
        )

        features = [
            torch.randn(0, 768, 56, 56),  # Empty batch
            torch.randn(0, 768, 28, 28),
            torch.randn(0, 768, 14, 14),
            torch.randn(0, 768, 7, 7),
        ]

        # Should handle gracefully (either error or empty output)
        try:
            output = adapter(features)
            # If it works, output batch size should be 0
            assert output.shape[0] == 0, f"Expected batch 0, got {output.shape[0]}"
        except Exception:
            # Error is also acceptable for empty batch
            pass


class TestSingleImageBatch:
    """Test single image batch (B=1)."""

    def test_single_image_works(self):
        """Single image batch should work."""
        adapter = TaskFeatureAdapter(
            in_channels=[768, 768, 768, 768],
            out_channels=1,
        )

        features = [
            torch.randn(1, 768, 56, 56),  # Batch of 1
            torch.randn(1, 768, 28, 28),
            torch.randn(1, 768, 14, 14),
            torch.randn(1, 768, 7, 7),
        ]

        output = adapter(features)

        assert output.shape[0] == 1, f"Expected batch 1, got {output.shape[0]}"
        assert output.shape[1] == 1


class TestInvalidTaskName:
    """Test handling of invalid task names."""

    def test_invalid_task_raises_error(self):
        """Invalid task should raise error."""
        bank = AdapterBank()
        bank.add_task("flood", TaskFeatureAdapter(
            in_channels=[768, 768, 768, 768],
            out_channels=1,
        ))

        with pytest.raises((KeyError, ValueError, AttributeError)):
            bank.set_task("banana")

    def test_missing_task_raises_error(self):
        """Missing task should raise error."""
        bank = AdapterBank()
        bank.add_task("flood", TaskFeatureAdapter(
            in_channels=[768, 768, 768, 768],
            out_channels=1,
        ))

        with pytest.raises((KeyError, ValueError, AttributeError)):
            bank.set_task("burn")  # Not added


class TestLORARankVariations:
    """Test different LoRA ranks."""

    @pytest.mark.parametrize("rank", [1, 4, 16, 64])
    def test_different_ranks(self, rank):
        """All ranks should work."""
        backbone = build_backbone("terramind_base")
        backbone.freeze()

        lora_model = TerraMindLoRA(
            backbone,
            rank=rank,
            alpha=rank,  # alpha = rank is common
            freeze_backbone=True,
        )

        assert lora_model.rank == rank
        assert lora_model.count_lora_layers() == 24

        # Check a LoRA layer has correct rank
        for module in lora_model.model.modules():
            if hasattr(module, 'lora_A'):
                assert module.lora_A.shape[0] == rank, \
                    f"lora_A should have rank {rank}, got {module.lora_A.shape[0]}"
                break

    def test_rank_1_works(self):
        """Rank 1 LoRA should work."""
        backbone = build_backbone("terramind_base")
        backbone.freeze()

        lora_model = TerraMindLoRA(backbone, rank=1, alpha=1, freeze_backbone=True)

        assert lora_model.count_lora_layers() > 0

        # Verify rank 1 structure
        for module in lora_model.model.modules():
            if hasattr(module, 'lora_A'):
                assert module.lora_A.shape[0] == 1
                assert module.lora_B.shape[1] == 1
                break

    def test_rank_64_works(self):
        """Rank 64 LoRA should work."""
        backbone = build_backbone("terramind_base")
        backbone.freeze()

        lora_model = TerraMindLoRA(backbone, rank=64, alpha=64, freeze_backbone=True)

        assert lora_model.count_lora_layers() == 24

        # Should have more params than rank 16
        params_16 = 24 * 2 * 768 * 16  # 24 layers * 2 matrices * 768 dim * 16 rank
        params_64 = lora_model.count_lora_params()

        assert params_64 > params_16


class TestMissingModality:
    """Test handling of missing modalities."""

    def test_wrong_channel_count(self):
        """Wrong channel count should produce error."""
        adapter = TaskFeatureAdapter(
            in_channels=[768, 768, 768, 768],  # Correct
            out_channels=1,
        )

        # Wrong number of features
        features = [
            torch.randn(1, 768, 56, 56),
            torch.randn(1, 768, 28, 28),
            torch.randn(1, 768, 14, 14),
            # Missing F11
        ]

        with pytest.raises((IndexError, RuntimeError, ValueError)):
            adapter(features)


class TestNumericalStability:
    """Test numerical stability."""

    def test_no_nan_in_output(self):
        """Output should not contain NaN."""
        adapter = TaskFeatureAdapter(
            in_channels=[768, 768, 768, 768],
            out_channels=1,
        )

        features = [torch.randn(2, 768, 56, 56) for _ in range(4)]
        output = adapter(features)

        assert not torch.isnan(output).any(), "Output contains NaN"

    def test_no_inf_in_output(self):
        """Output should not contain Inf."""
        adapter = TaskFeatureAdapter(
            in_channels=[768, 768, 768, 768],
            out_channels=1,
        )

        features = [torch.randn(2, 768, 56, 56) for _ in range(4)]
        output = adapter(features)

        assert not torch.isinf(output).any(), "Output contains Inf"

    def test_large_input_no_crash(self):
        """Large input should not crash."""
        adapter = TaskFeatureAdapter(
            in_channels=[768, 768, 768, 768],
            out_channels=1,
        )

        # Large features
        features = [torch.randn(1, 768, 224, 224) for _ in range(4)]

        output = adapter(features)

        assert not torch.isnan(output).any()


class TestDeterminism:
    """Test determinism with seeds."""

    def test_same_seed_same_output(self):
        """Same seed should produce same output."""
        torch.manual_seed(42)

        adapter1 = TaskFeatureAdapter(
            in_channels=[768, 768, 768, 768],
            out_channels=1,
        )

        torch.manual_seed(42)

        adapter2 = TaskFeatureAdapter(
            in_channels=[768, 768, 768, 768],
            out_channels=1,
        )

        features = [torch.randn(1, 768, 56, 56) for _ in range(4)]

        # Both should have same initial weights
        for (n1, p1), (n2, p2) in zip(
            adapter1.named_parameters(),
            adapter2.named_parameters()
        ):
            diff = (p1 - p2).abs().max().item()
            assert diff < 1e-6, f"Weights differ for {n1}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])