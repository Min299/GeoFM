"""tests/test_metadata_ablation.py

Tests for MetadataAblation.
"""
from geofm.experiments.metadata_ablation import (
    MetadataAblation,
)

from geofm.experiments.metadata_masker import (
    MetadataMasker,
)


def test_ablation():
    """Test basic ablation functionality."""
    ab = MetadataAblation(
        MetadataMasker()
    )

    out = ab.apply(
        {
            "lat": 10
        }
    )

    assert (
        out["lat"] == 0
    )


def test_ablation_with_dropper():
    """Test ablation with dropper strategy."""
    from geofm.experiments.metadata_dropper import MetadataDropper

    ab = MetadataAblation(
        MetadataDropper(drop_prob=0.0)
    )

    out = ab.apply({"lat": 10})

    assert out["lat"] == 10