"""tests/test_metadata_encoder.py

Tests for metadata encoder.
"""
import torch

from geofm.metadata.metadata_encoder import (
    MetadataEncoder,
)


def test_encoder():
    """Test basic encoding functionality."""
    encoder = MetadataEncoder()

    tensor = encoder.encode(
        {
            "latitude": 10,
            "longitude": 20,
            "resolution": 30,
        }
    )

    assert tensor.shape[0] == 3


def test_encoder_returns_tensor():
    """Test that encoder returns torch tensor."""
    encoder = MetadataEncoder()
    tensor = encoder.encode({})

    assert isinstance(tensor, torch.Tensor)


def test_encoder_default_values():
    """Test encoder uses defaults for missing fields."""
    encoder = MetadataEncoder()
    tensor = encoder.encode({})

    # Should have 3 values (lat, lon, resolution)
    assert tensor.shape[0] == 3
    # Default value is 0.0
    assert tensor.tolist() == [0.0, 0.0, 0.0]


def test_encoder_float32():
    """Test that encoder returns float32 tensor."""
    encoder = MetadataEncoder()
    tensor = encoder.encode({"latitude": 10})

    assert tensor.dtype == torch.float32