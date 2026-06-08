"""tests/test_task_dataset_router.py

Tests for task dataset router.
"""
import pytest
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    """Dummy dataset for testing."""

    def __init__(self, length=10):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {"image": idx, "label": idx % 3}


class TestTaskDatasetRouter:
    """Test TaskDatasetRouter class."""

    def test_init(self):
        """Router should initialize with datasets."""
        from geofm.data.task_dataset_router import TaskDatasetRouter

        datasets = {
            "flood": DummyDataset(10),
            "burn": DummyDataset(5),
        }

        router = TaskDatasetRouter(datasets)

        assert router.task_count() == 2

    def test_get_dataset(self):
        """get_dataset should return correct dataset."""
        from geofm.data.task_dataset_router import TaskDatasetRouter

        datasets = {
            "flood": DummyDataset(10),
            "burn": DummyDataset(5),
        }

        router = TaskDatasetRouter(datasets)

        flood_ds = router.get_dataset("flood")

        assert len(flood_ds) == 10

    def test_get_dataset_unknown(self):
        """get_dataset should raise for unknown task."""
        from geofm.data.task_dataset_router import TaskDatasetRouter

        router = TaskDatasetRouter({})

        with pytest.raises(KeyError):
            router.get_dataset("unknown")

    def test_available_tasks(self):
        """available_tasks should return task list."""
        from geofm.data.task_dataset_router import TaskDatasetRouter

        datasets = {
            "flood": DummyDataset(10),
            "burn": DummyDataset(5),
        }

        router = TaskDatasetRouter(datasets)

        tasks = router.available_tasks()

        assert "flood" in tasks
        assert "burn" in tasks

    def test_has_task(self):
        """has_task should check task existence."""
        from geofm.data.task_dataset_router import TaskDatasetRouter

        datasets = {"flood": DummyDataset(10)}

        router = TaskDatasetRouter(datasets)

        assert router.has_task("flood")
        assert not router.has_task("burn")

    def test_add_dataset(self):
        """add_dataset should add new task."""
        from geofm.data.task_dataset_router import TaskDatasetRouter

        datasets = {"flood": DummyDataset(10)}

        router = TaskDatasetRouter(datasets)
        router.add_dataset("burn", DummyDataset(5))

        assert router.has_task("burn")
        assert router.task_count() == 2

    def test_remove_dataset(self):
        """remove_dataset should remove task."""
        from geofm.data.task_dataset_router import TaskDatasetRouter

        datasets = {
            "flood": DummyDataset(10),
            "burn": DummyDataset(5),
        }

        router = TaskDatasetRouter(datasets)
        router.remove_dataset("burn")

        assert not router.has_task("burn")
        assert router.task_count() == 1

    def test_get_all_lengths(self):
        """get_all_lengths should return dataset lengths."""
        from geofm.data.task_dataset_router import TaskDatasetRouter

        datasets = {
            "flood": DummyDataset(10),
            "burn": DummyDataset(5),
        }

        router = TaskDatasetRouter(datasets)

        lengths = router.get_all_lengths()

        assert lengths["flood"] == 10
        assert lengths["burn"] == 5

    def test_get_total_samples(self):
        """get_total_samples should return sum."""
        from geofm.data.task_dataset_router import TaskDatasetRouter

        datasets = {
            "flood": DummyDataset(10),
            "burn": DummyDataset(5),
        }

        router = TaskDatasetRouter(datasets)

        total = router.get_total_samples()

        assert total == 15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])