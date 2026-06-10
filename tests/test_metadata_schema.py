"""tests/test_metadata_schema.py

Tests for metadata schema.
"""
from geofm.metadata.metadata_schema import (
    DEFAULT_METADATA_FIELDS,
)


def test_schema():
    """Test that default schema contains expected fields."""
    assert (
        "latitude"
        in DEFAULT_METADATA_FIELDS
    )
    assert (
        "longitude"
        in DEFAULT_METADATA_FIELDS
    )
    assert (
        "timestamp"
        in DEFAULT_METADATA_FIELDS
    )


def test_schema_has_sensor():
    """Test that schema includes sensor field."""
    assert "sensor" in DEFAULT_METADATA_FIELDS


def test_schema_has_platform():
    """Test that schema includes platform field."""
    assert "platform" in DEFAULT_METADATA_FIELDS


def test_schema_has_resolution():
    """Test that schema includes resolution field."""
    assert "resolution" in DEFAULT_METADATA_FIELDS


def test_schema_has_task():
    """Test that schema includes task field."""
    assert "task" in DEFAULT_METADATA_FIELDS