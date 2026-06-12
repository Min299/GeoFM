"""tests/test_checkpoint_manager.py

Tests for checkpoint manager.
"""
import pytest
import torch


class TestCheckpointManager:
    """Test CheckpointManager class."""

    def test_save_checkpoint(self, tmp_path):
        """Save checkpoint should create file."""
        from geofm.trainers.checkpoint_manager import CheckpointManager

        model = torch.nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())

        path = tmp_path / "test.pt"

        CheckpointManager.save(
            str(path),
            model,
            optimizer,
            epoch=5,
            metrics={"loss": 0.5},
        )

        assert path.exists()

    def test_load_checkpoint(self, tmp_path):
        """Load checkpoint should restore state."""
        from geofm.trainers.checkpoint_manager import CheckpointManager

        model = torch.nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())

        # Save
        path = tmp_path / "test.pt"
        CheckpointManager.save(str(path), model, optimizer, epoch=5)

        # Modify model
        for p in model.parameters():
            p.data.fill_(0)

        # Load
        metadata = CheckpointManager.load(str(path), model, optimizer)

        assert metadata["epoch"] == 5

    def test_save_best(self, tmp_path):
        """save_best should only save if better."""
        from geofm.trainers.checkpoint_manager import CheckpointManager

        model = torch.nn.Linear(10, 5)

        path = tmp_path / "test.pt"

        # First save should be best
        is_best = CheckpointManager.save_best(
            str(path),
            model,
            metrics={"loss": 0.5},
            current_metric=0.5,
            mode="min",
        )

        assert is_best
        assert (tmp_path / "test_best.pt").exists()

        # Second save with worse metric should not be best
        is_best = CheckpointManager.save_best(
            str(path),
            model,
            metrics={"loss": 0.8},
            current_metric=0.8,
            mode="min",
        )

        assert not is_best

    def test_list_checkpoints(self, tmp_path):
        """list_checkpoints should return all checkpoints."""
        from geofm.trainers.checkpoint_manager import CheckpointManager

        model = torch.nn.Linear(10, 5)

        # Create checkpoints
        (tmp_path / "ckpt_1.pt").touch()
        (tmp_path / "ckpt_2.pt").touch()
        (tmp_path / "other.txt").touch()

        checkpoints = CheckpointManager.list_checkpoints(str(tmp_path))

        assert len(checkpoints) == 2

    def test_get_latest_checkpoint(self, tmp_path):
        """get_latest_checkpoint should return most recent."""
        from geofm.trainers.checkpoint_manager import CheckpointManager

        # Create checkpoints with different timestamps
        (tmp_path / "ckpt_1.pt").touch()
        import time
        time.sleep(0.01)
        (tmp_path / "ckpt_2.pt").touch()

        latest = CheckpointManager.get_latest_checkpoint(str(tmp_path))

        assert latest is not None
        assert "ckpt_2.pt" in latest


if __name__ == "__main__":
    pytest.main([__file__, "-v"])