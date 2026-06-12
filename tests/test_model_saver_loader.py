"""tests/test_model_saver_loader.py

Tests for model saver and loader.
"""
import pytest
import torch
import torch.nn as nn


class TestModelSaver:
    """Test ModelSaver class."""

    def test_save(self, tmp_path):
        """save should create checkpoint file."""
        from geofm.checkpoint.model_saver import ModelSaver

        model = nn.Linear(10, 2)

        path = tmp_path / "model.pt"

        ModelSaver.save(model, path)

        assert path.exists()

    def test_save_checkpoint(self, tmp_path):
        """save_checkpoint should save with metadata."""
        from geofm.checkpoint.model_saver import ModelSaver

        model = nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters())

        path = tmp_path / "checkpoint.pt"

        ModelSaver.save_checkpoint(
            model,
            optimizer,
            path,
            epoch=5,
            metrics={"loss": 0.5},
        )

        assert path.exists()

        # Verify content
        checkpoint = torch.load(path)
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert checkpoint["epoch"] == 5


class TestModelLoader:
    """Test ModelLoader class."""

    def test_load(self, tmp_path):
        """load should restore model state."""
        from geofm.checkpoint.model_saver import ModelSaver
        from geofm.checkpoint.model_loader import ModelLoader

        model = nn.Linear(10, 2)

        path = tmp_path / "model.pt"

        ModelSaver.save(model, path)

        # Create new model
        new_model = nn.Linear(10, 2)

        # Fill with different values
        for p in new_model.parameters():
            p.data.fill_(0)

        # Load
        ModelLoader.load(new_model, path)

        # Model should have restored weights
        # (not all zeros)
        assert not torch.all(new_model.weight == 0).item()

    def test_load_checkpoint(self, tmp_path):
        """load_checkpoint should return dict."""
        from geofm.checkpoint.model_saver import ModelSaver
        from geofm.checkpoint.model_loader import ModelLoader

        model = nn.Linear(10, 2)

        path = tmp_path / "checkpoint.pt"

        ModelSaver.save_checkpoint(
            model,
            None,
            path,
            epoch=3,
            metrics={"accuracy": 0.9},
        )

        checkpoint = ModelLoader.load_checkpoint(path)

        assert checkpoint["epoch"] == 3
        assert checkpoint["metrics"]["accuracy"] == 0.9

    def test_load_nonexistent(self, tmp_path):
        """load should raise for nonexistent file."""
        from geofm.checkpoint.model_loader import ModelLoader

        model = nn.Linear(10, 2)

        with pytest.raises(FileNotFoundError):
            ModelLoader.load(model, tmp_path / "nonexistent.pt")


class TestSaveLoad:
    """Test save and load workflow."""

    def test_save_load_roundtrip(self, tmp_path):
        """Save and load should preserve weights."""
        from geofm.checkpoint.model_saver import ModelSaver
        from geofm.checkpoint.model_loader import ModelLoader

        model = nn.Linear(10, 2)

        # Set specific weights
        with torch.no_grad():
            model.weight.fill_(1.5)
            model.bias.fill_(0.5)

        path = tmp_path / "model.pt"

        ModelSaver.save(model, path)

        # Create new model
        new_model = nn.Linear(10, 2)

        ModelLoader.load(new_model, path)

        # Verify weights match
        assert torch.allclose(model.weight, new_model.weight)
        assert torch.allclose(model.bias, new_model.bias)

    def test_checkpoint_with_optimizer(self, tmp_path):
        """Checkpoint should preserve optimizer state."""
        from geofm.checkpoint.model_saver import ModelSaver
        from geofm.checkpoint.model_loader import ModelLoader

        model = nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters())

        # Train a bit to populate optimizer state
        x = torch.randn(5, 10)
        y = model(x)
        loss = y.mean()
        loss.backward()
        optimizer.step()

        path = tmp_path / "checkpoint.pt"

        ModelSaver.save_checkpoint(model, optimizer, path, epoch=1)

        # Load into new model and optimizer
        new_model = nn.Linear(10, 2)
        new_optimizer = torch.optim.Adam(new_model.parameters())

        # Load checkpoint and restore state
        checkpoint = ModelLoader.load_checkpoint(path)
        new_model.load_state_dict(checkpoint["model_state_dict"])
        new_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Optimizer state should be loaded
        assert len(new_optimizer.state) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])