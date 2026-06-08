"""tests/test_multitask_trainer.py

Tests for multi-task trainer.
"""
import pytest


class TestMultiTaskTrainer:
    """Test MultiTaskTrainer class."""

    def test_init(self):
        """Trainer should initialize with model, optimizer, criterion."""
        import torch
        import torch.nn as nn

        from geofm.training.multitask_trainer import MultiTaskTrainer

        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()

        trainer = MultiTaskTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            tasks=["flood", "burn"],
        )

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.criterion is not None
        assert len(trainer.tasks) == 2

    def test_get_step_count(self):
        """get_step_count should return step count."""
        import torch
        import torch.nn as nn

        from geofm.training.multitask_trainer import MultiTaskTrainer

        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()

        trainer = MultiTaskTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            tasks=["flood"],
        )

        assert trainer.get_step_count() == 0

    def test_reset_step_count(self):
        """reset_step_count should reset to 0."""
        import torch
        import torch.nn as nn

        from geofm.training.multitask_trainer import MultiTaskTrainer

        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()

        trainer = MultiTaskTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            tasks=["flood"],
        )

        # Simulate steps
        trainer.step_count = 10

        trainer.reset_step_count()

        assert trainer.get_step_count() == 0


class TestGradientMultiTaskTrainer:
    """Test GradientMultiTaskTrainer class."""

    def test_accumulation_steps(self):
        """GradientMultiTaskTrainer should have accumulation_steps."""
        import torch
        import torch.nn as nn

        from geofm.training.multitask_trainer import GradientMultiTaskTrainer

        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()

        trainer = GradientMultiTaskTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            tasks=["flood"],
            accumulation_steps=4,
        )

        assert trainer.accumulation_steps == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])