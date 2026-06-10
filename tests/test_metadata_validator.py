"""tests/test_metadata_validator.py

Tests for metadata validator.
"""
from geofm.metadata.metadata_validator import (
    MetadataValidator,
)


def test_validator():
    """Test basic validation with valid fields."""
    assert MetadataValidator.validate(
        {
            "latitude": 10,
            "longitude": 20,
        }
    )


def test_validator_with_all_fields():
    """Test validation with all default fields."""
    metadata = {
        "latitude": 40.0,
        "longitude": -74.0,
        "timestamp": "2024-01-01",
        "sensor": "S2",
        "platform": "Sentinel-2",
        "resolution": 10.0,
        "task": "flood",
    }
    assert MetadataValidator.validate(metadata) is True


def test_validator_unknown_field():
    """Test validation fails with unknown fields."""
    try:
        MetadataValidator.validate(
            {
                "latitude": 10,
                "unknown_field": 123,
            }
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "unknown_field" in str(e)