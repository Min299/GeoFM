"""tests/test_metadata_builder.py

Tests for metadata builder.
"""
from geofm.metadata.metadata_builder import (
    MetadataBuilder,
)


def test_builder():
    """Test basic builder functionality."""
    sample = (
        MetadataBuilder.from_dict(
            {
                "latitude": 10,
                "longitude": 20,
            }
        )
    )

    assert (
        sample.latitude == 10
    )
    assert (
        sample.longitude == 20
    )


def test_builder_all_fields():
    """Test builder with all fields."""
    data = {
        "latitude": 40.0,
        "longitude": -74.0,
        "timestamp": "2024-01-01",
        "sensor": "S2",
        "platform": "Sentinel-2",
        "resolution": 10.0,
        "task": "flood",
    }
    sample = MetadataBuilder.from_dict(data)

    assert sample.latitude == 40.0
    assert sample.longitude == -74.0
    assert sample.timestamp == "2024-01-01"
    assert sample.sensor == "S2"
    assert sample.platform == "Sentinel-2"
    assert sample.resolution == 10.0
    assert sample.task == "flood"


def test_builder_missing_fields():
    """Test builder with missing fields returns None."""
    sample = MetadataBuilder.from_dict({})

    assert sample.latitude is None
    assert sample.longitude is None
    assert sample.timestamp is None