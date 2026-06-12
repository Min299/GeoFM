"""tests/test_multitask_dataset.py

Tests for multi-task dataset.
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


class TestMultiTaskDataset:
    """Test MultiTaskDataset class."""

    def test_init(self):
        """MultiTaskDataset should initialize."""
        from geofm.data.multitask_dataset import MultiTaskDataset

        datasets = {
            "flood": DummyDataset(10),
            "burn": DummyDataset(10),
        }

        mt_dataset = MultiTaskDataset(datasets)

        assert mt_dataset.task_count() == 2
        assert "flood" in mt_dataset.tasks
        assert "burn" in mt_dataset.tasks

    def test_len(self):
        """__len__ should return total samples."""
        from geofm.data.multitask_dataset import MultiTaskDataset

        datasets = {
            "flood": DummyDataset(10),
            "burn": DummyDataset(10),
        }

        mt_dataset = MultiTaskDataset(datasets, cycle=True)

        # With cycle=True, length = min_len * num_tasks = 10 * 2 = 20
        assert len(mt_dataset) == 20

    def test_getitem(self):
        """__getitem__ should return sample with task."""
        from geofm.data.multitask_dataset import MultiTaskDataset

        datasets = {
            "flood": DummyDataset(10),
            "burn": DummyDataset(10),
        }

        mt_dataset = MultiTaskDataset(datasets, cycle=True)

        sample = mt_dataset[0]

        assert "task" in sample
        assert sample["task"] in ["flood", "burn"]

    def test_task_injection(self):
        """Sample should have task field."""
        from geofm.data.multitask_dataset import MultiTaskDataset

        datasets = {
            "flood": DummyDataset(5),
            "burn": DummyDataset(5),
        }

        mt_dataset = MultiTaskDataset(datasets, cycle=True)

        # Check multiple samples
        for idx in range(10):
            sample = mt_dataset[idx]
            assert "task" in sample
            assert sample["task"] in ["flood", "burn"]

    def test_available_tasks(self):
        """available_tasks should return task list."""
        from geofm.data.multitask_dataset import MultiTaskDataset

        datasets = {
            "flood": DummyDataset(10),
            "burn": DummyDataset(10),
            "lulc": DummyDataset(10),
        }

        mt_dataset = MultiTaskDataset(datasets)

        tasks = mt_dataset.available_tasks()

        assert len(tasks) == 3
        assert "flood" in tasks
        assert "burn" in tasks
        assert "lulc" in tasks

    def test_concat_mode(self):
        """Concat mode should concatenate samples."""
        from geofm.data.multitask_dataset import MultiTaskDataset

        datasets = {
            "flood": DummyDataset(10),
            "burn": DummyDataset(5),
        }

        mt_dataset = MultiTaskDataset(datasets, cycle=False)

        # Total = 10 + 5 = 15
        assert len(mt_dataset) == 15

    def test_cycle_mode(self):
        """Cycle mode should cycle evenly."""
        from geofm.data.multitask_dataset import MultiTaskDataset

        datasets = {
            "flood": DummyDataset(10),
            "burn": DummyDataset(5),
        }

        mt_dataset = MultiTaskDataset(datasets, cycle=True)

        # min(10, 5) * 2 = 10
        assert len(mt_dataset) == 10

    def test_get_task_dataset(self):
        """get_task_dataset should return correct dataset."""
        from geofm.data.multitask_dataset import MultiTaskDataset

        datasets = {
            "flood": DummyDataset(10),
            "burn": DummyDataset(5),
        }

        mt_dataset = MultiTaskDataset(datasets)

        flood_ds = mt_dataset.get_task_dataset("flood")
        assert len(flood_ds) == 10

    def test_task_sample_counts(self):
        """get_task_sample_counts should return counts."""
        from geofm.data.multitask_dataset import MultiTaskDataset

        datasets = {
            "flood": DummyDataset(10),
            "burn": DummyDataset(5),
        }

        mt_dataset = MultiTaskDataset(datasets)

        counts = mt_dataset.get_task_sample_counts()

        assert counts["flood"] == 10
        assert counts["burn"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])