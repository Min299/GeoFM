"""tests/test_metadata_masker.py

Tests for MetadataMasker.
"""
from geofm.experiments.metadata_masker import (
    MetadataMasker,
)


def test_masker():
    """Test basic masker functionality."""
    m = MetadataMasker()

    out = m(
        {
            "lat": 10
        }
    )

    assert (
        out["lat"] == 0
    )


def test_masker_custom_value():
    """Test masker with custom mask value."""
    m = MetadataMasker(
        mask_value=-1
    )

    out = m({"lat": 10})

    assert out["lat"] == -1


def test_masker_all_keys():
    """Test that masker preserves all keys."""
    m = MetadataMasker()

    out = m({
        "lat": 10,
        "lon": 20,
        "timestamp": "2024",
    })

    assert "lat" in out
    assert "lon" in out
    assert "timestamp" in out
    assert all(v == 0 for k, v in out.items() if isinstance(v, (int, float)))