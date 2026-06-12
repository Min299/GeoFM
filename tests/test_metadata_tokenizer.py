"""tests/test_metadata_tokenizer.py

Tests for MetadataTokenizer.
"""
import torch

from geofm.models.metadata.metadata_tokenizer import (
    MetadataTokenizer,
)


def test_tokenizer():
    """Test basic tokenization functionality."""
    tok = MetadataTokenizer(
        3,
        128,
    )

    x = torch.randn(
        2,
        3,
    )

    y = tok(x)

    assert y.shape == (
        2,
        1,
        128,
    )


def test_tokenizer_single_sample():
    """Test tokenization with single sample."""
    tok = MetadataTokenizer(3, 128)
    x = torch.randn(3)
    y = tok(x)

    assert y.shape == (1, 128)


def test_tokenizer_callable():
    """Test that tokenizer is callable."""
    tok = MetadataTokenizer(3, 128)
    x = torch.randn(2, 3)

    # __call__ should work
    y1 = tok(x)
    y2 = tok.forward(x)

    assert torch.equal(y1, y2)