"""tests/test_segmentation_model.py

Tests for SegmentationModel.
"""
from geofm.models.tasks.segmentation_model import (
    SegmentationModel,
)


def test_segmentation_exists():
    """Test SegmentationModel class exists."""
    assert (
        SegmentationModel
        is not None
    )


def test_segmentation_init():
    """Test SegmentationModel initialization."""
    model = SegmentationModel(
        backbone=None,
        decoder=None,
    )

    assert model is not None
    assert model.backbone is None
    assert model.decoder is None