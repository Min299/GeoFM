"""tests/test_collate.py

Tests for collate functions.
"""
import pytest
import torch


class TestGeofmCollate:
    """Test geofm_collate function."""

    def test_collate_stacking(self):
        """geofm_collate should stack tensors."""
        from geofm.data.collate import geofm_collate

        batch = [
            {"image": torch.randn(3, 32, 32), "label": 0},
            {"image": torch.randn(3, 32, 32), "label": 1},
        ]

        result = geofm_collate(batch)

        assert result["image"].shape == (2, 3, 32, 32)
        assert result["label"].shape == (2,)

    def test_collate_strings(self):
        """geofm_collate should keep strings as list."""
        from geofm.data.collate import geofm_collate

        batch = [
            {"image": torch.randn(3, 32, 32), "path": "a.jpg"},
            {"image": torch.randn(3, 32, 32), "path": "b.jpg"},
        ]

        result = geofm_collate(batch)

        assert result["path"] == ["a.jpg", "b.jpg"]

    def test_collate_empty_batch(self):
        """geofm_collate should handle empty batch."""
        from geofm.data.collate import geofm_collate

        result = geofm_collate([])

        assert result == {}

    def test_collate_multitask(self):
        """geofm_collate should handle task field."""
        from geofm.data.collate import geofm_collate

        batch = [
            {"image": torch.randn(3, 32, 32), "task": "flood"},
            {"image": torch.randn(3, 32, 32), "task": "burn"},
        ]

        result = geofm_collate(batch)

        # Task should be preserved as list
        assert result["task"] == ["flood", "burn"]


class TestSegmentationCollate:
    """Test segmentation_collate function."""

    def test_segmentation_collate(self):
        """segmentation_collate should handle image and mask."""
        from geofm.data.collate import segmentation_collate

        batch = [
            {
                "image": torch.randn(3, 64, 64),
                "mask": torch.randint(0, 5, (64, 64)),
            },
            {
                "image": torch.randn(3, 64, 64),
                "mask": torch.randint(0, 5, (64, 64)),
            },
        ]

        result = segmentation_collate(batch)

        assert result["image"].shape == (2, 3, 64, 64)
        assert result["mask"].shape == (2, 64, 64)


class TestMultitaskCollate:
    """Test multitask_collate function."""

    def test_multitask_collate(self):
        """multitask_collate should preserve task info."""
        from geofm.data.collate import multitask_collate

        batch = [
            {"image": torch.randn(3, 32, 32), "task": "flood"},
            {"image": torch.randn(3, 32, 32), "task": "burn"},
        ]

        result = multitask_collate(batch)

        assert result["task"] == ["flood", "burn"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])