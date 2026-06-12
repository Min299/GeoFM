"""tests/test_metadata_registry.py

Tests for metadata registry.
"""
from geofm.metadata.metadata_registry import (
    MetadataRegistry,
)


def test_registry():
    """Test basic registry functionality."""
    reg = MetadataRegistry()

    reg.register(
        "flood",
        [
            "latitude",
            "longitude",
        ],
    )

    assert (
        "flood"
        in reg.datasets()
    )


def test_get_fields():
    """Test getting fields for registered dataset."""
    reg = MetadataRegistry()
    reg.register("flood", ["latitude", "longitude"])

    fields = reg.get_fields("flood")
    assert "latitude" in fields
    assert "longitude" in fields


def test_get_fields_unknown_dataset():
    """Test getting fields for unknown dataset returns empty list."""
    reg = MetadataRegistry()

    fields = reg.get_fields("unknown")
    assert fields == []


def test_datasets_empty():
    """Test that empty registry has no datasets."""
    reg = MetadataRegistry()
    assert reg.datasets() == []