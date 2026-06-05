"""Tests for dataset registry and builder."""
import pytest
import tempfile
import os

pytest.importorskip("torch")

from geofm.datasets.registry import (
    DATASET_REGISTRY,
    register_dataset,
    list_registered_tasks,
    is_registered,
)
from geofm.datasets.builder import build_dataset, DatasetBuildError
from geofm.datasets.base_dataset import BaseGeoDataset


class DummyDataset(BaseGeoDataset):
    """Dummy dataset for testing."""

    def __init__(self, root_dir, task="dummy"):
        super().__init__(root_dir, task)
        self.samples = [f"sample_{i}" for i in range(5)]

    def __len__(self):
        return len(self.samples)

    def load_image(self, idx):
        import torch
        return torch.zeros(3, 64, 64)

    def load_label(self, idx):
        import torch
        return torch.zeros(64, 64, dtype=torch.long)

    def load_metadata(self, idx):
        return {"lat_sin": 0.0, "lat_cos": 1.0}


class TestRegistry:
    def test_flood_is_registered(self):
        assert is_registered("flood")

    def test_list_registered_tasks(self):
        tasks = list_registered_tasks()
        assert "flood" in tasks
        assert isinstance(tasks, list)

    def test_register_decorator(self):
        @register_dataset("dummy")
        class NewDataset(BaseGeoDataset):
            def __len__(self):
                return 0

            def load_image(self, idx):
                pass

            def load_label(self, idx):
                pass

            def load_metadata(self, idx):
                pass

        assert is_registered("dummy")
        assert "dummy" in list_registered_tasks()


class TestBuilder:
    def test_build_unknown_task_raises(self):
        with pytest.raises(DatasetBuildError) as exc_info:
            build_dataset(task="unknown_task", root_dir="/tmp")
        assert "not registered" in str(exc_info.value)

    def test_build_flood_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy structure
            img_dir = os.path.join(tmpdir, "images")
            lbl_dir = os.path.join(tmpdir, "labels")
            os.makedirs(img_dir)
            os.makedirs(lbl_dir)
            
            # Should not raise even without actual data
            dataset = build_dataset(task="flood", root_dir=tmpdir)
            assert dataset is not None
