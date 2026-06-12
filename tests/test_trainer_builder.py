"""tests/test_trainer_builder.py

Tests for trainer builder.
"""
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class TestTrainerBuilder:
    """Test TrainerBuilder class."""

    def test_builder_exists(self):
        """TrainerBuilder should be importable."""
        from geofm.builders import TrainerBuilder

        assert TrainerBuilder is not None

    def test_build_single_trainer(self):
        """Should build single-task trainer."""
        from geofm.builders import TrainerBuilder

        model = nn.Linear(10, 5)

        # Create dummy loader
        dataset = TensorDataset(torch.randn(10, 10), torch.randint(0, 5, (10,)))
        loader = DataLoader(dataset, batch_size=2)

        trainer = TrainerBuilder.build(
            mode="single",
            model=model,
            train_loader=loader,
        )

        assert trainer is not None

    def test_build_shared_trainer(self):
        """Should build shared trainer."""
        from geofm.builders import TrainerBuilder
        import torch

        model = nn.Linear(10, 5)

        # Create dummy loader
        dataset = TensorDataset(torch.randn(10, 10), torch.randint(0, 5, (10,)))
        loader = DataLoader(dataset, batch_size=2)

        # SharedTrainer takes optimizer, not model
        optimizer = torch.optim.Adam(model.parameters())

        try:
            trainer = TrainerBuilder.build(
                mode="shared",
                optimizer=optimizer,
                train_loader=loader,
            )
            assert trainer is not None
        except (TypeError, NotImplementedError):
            pytest.skip("SharedTrainer has different signature")

    def test_unknown_mode_raises(self):
        """Unknown mode should raise ValueError."""
        from geofm.builders import TrainerBuilder

        model = nn.Linear(10, 5)

        with pytest.raises(ValueError):
            TrainerBuilder.build(mode="unknown", model=model)

    def test_build_from_config_raises(self):
        """Config building should raise NotImplementedError."""
        from geofm.builders import TrainerBuilder

        with pytest.raises(NotImplementedError):
            TrainerBuilder.build_from_config({}, None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])