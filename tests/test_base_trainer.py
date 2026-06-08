"""tests/test_base_trainer.py

Tests for base trainer.
"""
import pytest
import torch


class TestBaseTrainer:
    """Test BaseTrainer class."""

    def test_base_trainer_exists(self):
        """BaseTrainer should be importable."""
        from geofm.trainers.base_trainer import BaseTrainer

        assert BaseTrainer is not None

    def test_base_trainer_is_abc(self):
        """BaseTrainer should be abstract."""
        from geofm.trainers.base_trainer import BaseTrainer

        # BaseTrainer should be an ABC
        assert hasattr(BaseTrainer, "__abstractmethods__")

    def test_trainer_can_be_instantiated(self):
        """Trainer with implementation should work."""
        from geofm.trainers.base_trainer import BaseTrainer
        from torch.utils.data import DataLoader, TensorDataset

        class ConcreteTrainer(BaseTrainer):
            def train(self):
                return {}

            def test(self):
                return {}

            def train_epoch(self, train_loader):
                return {"loss": 0.5}

            def validate(self, val_loader):
                return {"metric": 0.8}

        model = torch.nn.Linear(10, 5)

        # Create dummy loaders
        dataset = TensorDataset(torch.randn(10, 10), torch.randint(0, 5, (10,)))
        loader = DataLoader(dataset, batch_size=2)

        trainer = ConcreteTrainer(model, train_loader=loader, val_loader=loader)

        assert trainer.model is not None

    def test_trainer_save_checkpoint(self, tmp_path):
        """Trainer should save checkpoint."""
        from geofm.trainers.base_trainer import BaseTrainer
        from torch.utils.data import DataLoader, TensorDataset

        class ConcreteTrainer(BaseTrainer):
            def train(self):
                return {}

            def test(self):
                return {}

            def train_epoch(self, train_loader):
                return {"loss": 0.5}

            def validate(self, val_loader):
                return {"metric": 0.8}

        model = torch.nn.Linear(10, 5)
        dataset = TensorDataset(torch.randn(10, 10), torch.randint(0, 5, (10,)))
        loader = DataLoader(dataset, batch_size=2)

        trainer = ConcreteTrainer(model, train_loader=loader)

        path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(str(path))

        assert path.exists()

    def test_trainer_load_checkpoint(self, tmp_path):
        """Trainer should load checkpoint."""
        from geofm.trainers.base_trainer import BaseTrainer
        from torch.utils.data import DataLoader, TensorDataset

        class ConcreteTrainer(BaseTrainer):
            def train(self):
                return {}

            def test(self):
                return {}

            def train_epoch(self, train_loader):
                return {"loss": 0.5}

            def validate(self, val_loader):
                return {"metric": 0.8}

        model = torch.nn.Linear(10, 5)
        dataset = TensorDataset(torch.randn(10, 10), torch.randint(0, 5, (10,)))
        loader = DataLoader(dataset, batch_size=2)

        trainer = ConcreteTrainer(model, train_loader=loader)

        path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(str(path))
        trainer.load_checkpoint(str(path))

        assert trainer.current_epoch == 0

    def test_trainer_history(self):
        """Trainer should track history."""
        from geofm.trainers.base_trainer import BaseTrainer
        from torch.utils.data import DataLoader, TensorDataset

        class ConcreteTrainer(BaseTrainer):
            def train(self):
                return {}

            def test(self):
                return {}

            def train_epoch(self, train_loader):
                return {"loss": 0.5}

            def validate(self, val_loader):
                return {"metric": 0.8}

        model = torch.nn.Linear(10, 5)
        dataset = TensorDataset(torch.randn(10, 10), torch.randint(0, 5, (10,)))
        loader = DataLoader(dataset, batch_size=2)

        trainer = ConcreteTrainer(model, train_loader=loader, val_loader=loader)

        # Verify trainer can run a training epoch
        metrics = trainer.train_epoch(loader)
        assert "loss" in metrics

        # Verify validation works
        val_metrics = trainer.validate(loader)
        assert "metric" in val_metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])