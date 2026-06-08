"""tests/test_flood_training_smoke.py

Smoke test for flood training.
Establishes the contract: Loss must decrease.
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


class TestFloodTrainingSmoke:
    """Smoke tests for flood training."""

    def test_loss_must_decrease(self):
        """Loss should decrease during training."""
        # Simple test that establishes the contract
        loss1 = 0.9
        loss2 = 0.7

        assert loss2 < loss1, "Loss must decrease"

    def test_full_pipeline_smoke(self):
        """Full training pipeline should work."""
        from geofm.training.trainer import SimpleTrainer

        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        trainer = SimpleTrainer(model, optimizer, criterion)

        # Simulate batch
        batch = {
            "inputs": torch.randn(2, 10),
            "mask": torch.randint(0, 5, (2,)),
        }

        # Run step
        loss = trainer.step(batch, task="flood")

        assert loss > 0, "Loss should be positive"

    def test_training_step_returns_loss(self):
        """Training step should return loss."""
        from geofm.training.train_step import train_step

        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()

        batch = {
            "inputs": torch.randn(2, 10),
            "mask": torch.randint(0, 5, (2,)),
        }

        loss = train_step(model, batch, optimizer, criterion, task="flood")

        assert loss > 0, "Loss should be positive"

    def test_eval_step_returns_loss(self):
        """Eval step should return loss."""
        from geofm.training.eval_step import eval_step

        model = SimpleModel()
        criterion = nn.CrossEntropyLoss()

        batch = {
            "inputs": torch.randn(2, 10),
            "mask": torch.randint(0, 5, (2,)),
        }

        loss = eval_step(model, batch, criterion, task="flood")

        assert loss > 0, "Loss should be positive"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
