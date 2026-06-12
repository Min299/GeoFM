"""tests/test_checkpoint_roundtrip.py

Tier 4: Experiment System Tests

Verify:
- Checkpoint save/load works correctly
- Model produces same output after save/load
"""
import pytest
import torch
import torch.nn as nn
import tempfile
import os

from geofm.trainers import CheckpointManager
from geofm.models.peft import TerraMindLoRA
from geofm.models.backbones import build_backbone


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class TestCheckpointRoundtrip:
    """Test checkpoint save and load."""

    @pytest.fixture
    def model(self):
        """Create model."""
        torch.manual_seed(42)
        return SimpleModel()

    @pytest.fixture
    def optimizer(self, model):
        """Create optimizer."""
        return torch.optim.Adam(model.parameters(), lr=1e-3)

    @pytest.fixture
    def dummy_input(self):
        """Create dummy input."""
        return torch.randn(1, 3, 32, 32)

    def test_save_and_load_basic(self, model, optimizer, dummy_input):
        """Basic save/load should preserve model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")

            # Save
            CheckpointManager.save(path, model, optimizer)

            # Load
            loaded_model = SimpleModel()
            CheckpointManager.load(path, loaded_model)

            # Verify outputs match
            with torch.no_grad():
                out1 = model(dummy_input)
                out2 = loaded_model(dummy_input)

            diff = (out1 - out2).abs().max().item()
            assert diff < 1e-6, f"Outputs differ after load: {diff}"

    def test_save_with_optimizer(self, model, optimizer, dummy_input):
        """Save/load should preserve optimizer state."""
        # Train one step
        output = model(dummy_input)
        loss = output.mean()
        loss.backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")

            # Save with optimizer
            CheckpointManager.save(path, model, optimizer)

            # Load
            loaded_model = SimpleModel()
            loaded_optimizer = torch.optim.Adam(loaded_model.parameters(), lr=1e-3)

            metadata = CheckpointManager.load(path, loaded_model, loaded_optimizer)

            # Verify optimizer state was loaded
            assert loaded_optimizer.state != {}, "Optimizer state not loaded"

    def test_predictions_match_after_reload(self, model, optimizer, dummy_input):
        """Predictions should match before and after checkpoint."""
        # Run forward pass
        with torch.no_grad():
            output_before = model(dummy_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")

            # Save and load
            CheckpointManager.save(path, model)
            CheckpointManager.load(path, model)

            # Run again
            with torch.no_grad():
                output_after = model(dummy_input)

        # Outputs should be identical
        diff = (output_before - output_after).abs().max().item()
        assert diff < 1e-6, f"Predictions changed after reload: {diff}"

    def test_load_nonexistent_file(self, model):
        """Loading nonexistent file should raise error."""
        with pytest.raises(OSError):
            CheckpointManager.load("/nonexistent/path/checkpoint.pt", model)


class TestCheckpointWithLoRA:
    """Test checkpoint with LoRA model."""

    @pytest.fixture
    def lora_model(self):
        """Create LoRA model."""
        backbone = build_backbone("terramind_base")
        backbone.freeze()

        lora_model = TerraMindLoRA(backbone, rank=8, alpha=8, freeze_backbone=True)
        return lora_model

    def test_lora_checkpoint_roundtrip(self, lora_model):
        """LoRA model should save and load correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "lora_model.pt")

            # Save
            CheckpointManager.save(path, lora_model)

            # Load into new model
            backbone2 = build_backbone("terramind_base")
            backbone2.freeze()
            lora_model2 = TerraMindLoRA(backbone2, rank=8, alpha=8, freeze_backbone=True)

            CheckpointManager.load(path, lora_model2)

            # LoRA params should match
            params1 = {
                k: v
                for k, v in lora_model.state_dict().items()
                if 'lora' in k
            }
            params2 = {
                k: v
                for k, v in lora_model2.state_dict().items()
                if 'lora' in k
            }

            for key in params1:
                diff = (params1[key] - params2[key]).abs().max().item()
                assert diff < 1e-6, f"LoRA param {key} differs after reload"


class TestCheckpointMetadata:
    """Test checkpoint metadata handling."""

    def test_save_with_metrics(self):
        """Should save metrics with checkpoint."""
        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")

            metrics = {"loss": 0.5, "accuracy": 0.9, "epoch": 10}

            CheckpointManager.save(path, model, metrics=metrics)

            # Load
            loaded = torch.load(path, map_location="cpu")

            assert "metrics" in loaded
            assert loaded["metrics"]["loss"] == 0.5
            assert loaded["metrics"]["epoch"] == 10

    def test_save_with_epoch(self):
        """Should save epoch number."""
        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")

            CheckpointManager.save(path, model, epoch=42)

            # Load
            loaded = torch.load(path, map_location="cpu")

            assert loaded.get("epoch") == 42


class TestSaveBestCheckpoint:
    """Test best checkpoint saving."""

    def test_save_best_minimizes(self):
        """Should save when metric improves (min mode)."""
        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            best_path = os.path.join(tmpdir, "best.pt")

            # First save
            saved1 = CheckpointManager.save_best(
                best_path, model,
                metrics={"loss": 1.0},
                current_metric=1.0,
                mode="min"
            )
            assert saved1 is True, "First save should succeed"

            # Worse metric
            saved2 = CheckpointManager.save_best(
                best_path, model,
                metrics={"loss": 2.0},
                current_metric=2.0,
                mode="min"
            )
            assert saved2 is False, "Should not save worse metric"

            # Better metric
            saved3 = CheckpointManager.save_best(
                best_path, model,
                metrics={"loss": 0.5},
                current_metric=0.5,
                mode="min"
            )
            assert saved3 is True, "Should save better metric"

    def test_save_best_maximizes(self):
        """Should save when metric improves (max mode)."""
        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            best_path = os.path.join(tmpdir, "best.pt")

            # First save (accuracy 0.5 is best so far)
            saved1 = CheckpointManager.save_best(
                best_path, model,
                metrics={"accuracy": 0.5},
                current_metric=0.5,  # This becomes the best
                mode="max"
            )
            assert saved1 is True

            # Worse metric (0.3 < 0.5, not an improvement)
            saved2 = CheckpointManager.save_best(
                best_path, model,
                metrics={"accuracy": 0.3},
                current_metric=0.3,
                mode="max"
            )
            assert saved2 is False

            # Better metric (0.8 > 0.5, is an improvement)
            saved3 = CheckpointManager.save_best(
                best_path, model,
                metrics={"accuracy": 0.8},
                current_metric=0.8,
                mode="max"
            )
            assert saved3 is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])