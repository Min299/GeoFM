"""tests/test_dataset_factory.py

Tests for dataset factory.
"""
import pytest


class TestDatasetFactory:
    """Test DatasetFactory class."""

    def test_register(self):
        """register should add dataset to registry."""
        from geofm.data.dataset_factory import DatasetFactory

        class DummyDataset:
            pass

        DatasetFactory.register("dummy", DummyDataset)

        assert DatasetFactory.is_registered("dummy")

        # Clean up
        DatasetFactory.unregister("dummy")

    def test_build(self):
        """build should create dataset instance."""
        from geofm.data.dataset_factory import DatasetFactory

        class TestDataset:
            def __init__(self, split):
                self.split = split

        DatasetFactory.register("test", TestDataset)

        dataset = DatasetFactory.build("test", split="train")

        assert dataset.split == "train"

        # Clean up
        DatasetFactory.unregister("test")

    def test_build_unknown(self):
        """build should raise for unknown dataset."""
        from geofm.data.dataset_factory import DatasetFactory

        with pytest.raises(ValueError):
            DatasetFactory.build("unknown_dataset")

    def test_available(self):
        """available should return list of names."""
        from geofm.data.dataset_factory import DatasetFactory

        class DummyDataset:
            pass

        DatasetFactory.register("test1", DummyDataset)
        DatasetFactory.register("test2", DummyDataset)

        available = DatasetFactory.available()

        assert "test1" in available
        assert "test2" in available

        # Clean up
        DatasetFactory.unregister("test1")
        DatasetFactory.unregister("test2")

    def test_unregister(self):
        """unregister should remove dataset."""
        from geofm.data.dataset_factory import DatasetFactory

        class DummyDataset:
            pass

        DatasetFactory.register("test", DummyDataset)
        assert DatasetFactory.is_registered("test")

        DatasetFactory.unregister("test")
        assert not DatasetFactory.is_registered("test")

    def test_clear(self):
        """clear should remove all datasets."""
        from geofm.data.dataset_factory import DatasetFactory

        class DummyDataset:
            pass

        DatasetFactory.register("test1", DummyDataset)
        DatasetFactory.register("test2", DummyDataset)

        DatasetFactory.clear()

        assert len(DatasetFactory.available()) == 0

    def test_register_decorator(self):
        """register_dataset decorator should work."""
        from geofm.data.dataset_factory import register_dataset, DatasetFactory

        @register_dataset("decorated")
        class DecoratedDataset:
            pass

        assert DatasetFactory.is_registered("decorated")

        # Clean up
        DatasetFactory.unregister("decorated")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])