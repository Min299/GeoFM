"""tests/test_pipeline.py

Tests for GeoFMPipeline.
"""
from geofm.integration.pipeline import (
    GeoFMPipeline,
)


def test_pipeline():
    """Test basic pipeline initialization."""
    pipe = GeoFMPipeline(
        model=None
    )

    assert pipe is not None


def test_pipeline_with_components():
    """Test pipeline with fusion and encoder."""
    pipe = GeoFMPipeline(
        model=None,
        fusion=None,
        metadata_encoder=None,
    )

    assert pipe.model is None
    assert pipe.fusion is None
    assert pipe.metadata_encoder is None