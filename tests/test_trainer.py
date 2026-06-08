"""tests/test_trainer.py

Tests for trainer.
"""
import pytest
import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x, task_name=None):
        return self.linear(x)


class TestTrainer:
    """Test trainer."""

    def test_trainer_init(self):
        """Trainer should initialize."""
        from geofm.training.trainer import Trainer, SimpleTrainer

        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()

        trainer = Trainer(model, optimizer, criterion)
        assert trainer is not None

        simple_trainer = SimpleTrainer(model, optimizer, criterion)
        assert simple_trainer is not None

    def test_simple_trainer_step(self):
        """SimpleTrainer.step should work."""
        from geofm.training.trainer import SimpleTrainer

        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()

        trainer = SimpleTrainer(model, optimizer, criterion)

        batch = {
            "inputs": torch.randn(2, 10),
            "mask": torch.randint(0, 5, (2,)),
        }

        loss = trainer.step(batch)

        assert loss > 0, "Loss should be positive"

    def test_trainer_train_step(self):
        """Trainer.train_step should work."""
        from geofm.training.trainer import Trainer

        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()

        trainer = Trainer(model, optimizer, criterion)

        batch = {
            "inputs": torch.randn(2, 10),
            "mask": torch.randint(0, 5, (2,)),
        }

        loss = trainer.train_step(batch)

        assert loss > 0, "Loss should be positive"

    def test_trainer_eval_step(self):
        """Trainer.eval_step should work."""
        from geofm.training.trainer import Trainer

        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()

        trainer = Trainer(model, optimizer, criterion)

        batch = {
            "inputs": torch.randn(2, 10),
            "mask": torch.randint(0, 5, (2,)),
        }

        loss = trainer.eval_step(batch)

        assert loss > 0, "Loss should be positive"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
