"""tests/test_metadata_projector.py

Tests for MetadataProjector.
"""
import torch

from geofm.models.metadata.metadata_projector import (
    MetadataProjector,
)


def test_projector():
    """Test basic projection functionality."""
    model = MetadataProjector(
        3,
        128,
    )

    x = torch.randn(
        2,
        3,
    )

    y = model(x)

    assert y.shape == (
        2,
        128,
    )


def test_projector_single_sample():
    """Test projection with single sample."""
    model = MetadataProjector(3, 128)
    x = torch.randn(3)
    y = model(x)

    assert y.shape == (128,)


def test_projector_trainable():
    """Test that projector parameters are trainable."""
    model = MetadataProjector(3, 128)

    params = list(model.parameters())
    assert len(params) > 0
    assert all(p.requires_grad for p in params)