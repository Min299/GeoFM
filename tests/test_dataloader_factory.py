"""tests/test_dataloader_factory.py

Tests for dataloader factory.
"""
import pytest
import torch
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    """Dummy dataset for testing."""

    def __init__(self, length=10):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            "image": torch.randn(3, 32, 32),
            "label": idx % 3,
        }


class TestDataLoaderFactory:
    """Test DataLoaderFactory class."""

    def test_build(self):
        """build should create DataLoader."""
        from geofm.data.dataloader_factory import DataLoaderFactory

        dataset = DummyDataset(10)

        loader = DataLoaderFactory.build(dataset, batch_size=4)

        assert loader.batch_size == 4

    def test_build_train_loader(self):
        """build_train_loader should create loader with correct settings."""
        from geofm.data.dataloader_factory import DataLoaderFactory

        dataset = DummyDataset(20)

        loader = DataLoaderFactory.build_train_loader(dataset, batch_size=4)

        assert loader.batch_size == 4
        assert loader.drop_last is True

    def test_build_val_loader(self):
        """build_val_loader should create loader with correct settings."""
        from geofm.data.dataloader_factory import DataLoaderFactory

        dataset = DummyDataset(20)

        loader = DataLoaderFactory.build_val_loader(dataset, batch_size=4)

        assert loader.batch_size == 4
        assert loader.drop_last is False

    def test_build_test_loader(self):
        """build_test_loader should create loader with correct settings."""
        from geofm.data.dataloader_factory import DataLoaderFactory

        dataset = DummyDataset(20)

        loader = DataLoaderFactory.build_test_loader(dataset, batch_size=4)

        assert loader.batch_size == 4
        assert loader.drop_last is False

    def test_create_dataloader(self):
        """create_dataloader should work."""
        from geofm.data.dataloader_factory import create_dataloader

        dataset = DummyDataset(10)

        loader = create_dataloader(dataset, batch_size=4)

        assert loader.batch_size == 4

    def test_iteration(self):
        """DataLoader should iterate correctly."""
        from geofm.data.dataloader_factory import DataLoaderFactory

        dataset = DummyDataset(10)

        loader = DataLoaderFactory.build(dataset, batch_size=4)

        batches = list(loader)

        # 10 samples / 4 batch_size = 2 full batches + 1 partial
        assert len(batches) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])