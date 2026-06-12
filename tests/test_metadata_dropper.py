"""tests/test_metadata_dropper.py

Tests for MetadataDropper.
"""
from geofm.experiments.metadata_dropper import (
    MetadataDropper,
)


def test_dropper():
    """Test basic dropper functionality."""
    d = MetadataDropper(
        drop_prob=1.0
    )

    out = d(
        {
            "lat": 10
        }
    )

    assert out == {}


def test_dropper_no_drop():
    """Test dropper with 0 probability."""
    d = MetadataDropper(
        drop_prob=0.0
    )

    out = d({"lat": 10})

    assert out == {"lat": 10}


def test_dropper_preserves_metadata():
    """Test that dropper preserves metadata when not dropped."""
    d = MetadataDropper(
        drop_prob=0.0
    )

    metadata = {"lat": 10, "lon": 20}
    out = d(metadata)

    assert out == metadata