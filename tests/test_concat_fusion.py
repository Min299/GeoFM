"""tests/test_concat_fusion.py

Tests for ConcatFusion.
"""
import torch

from geofm.models.fusion.concat_fusion import (
    ConcatFusion,
)


def test_concat():
    """Test concatenation fusion."""
    fusion = ConcatFusion()

    img = torch.randn(
        2,
        196,
        128,
    )

    meta = torch.randn(
        2,
        1,
        128,
    )

    out = fusion(
        img,
        meta,
    )

    assert out.shape[1] == 197


def test_concat_preserves_dim():
    """Test that concat preserves embedding dimension."""
    fusion = ConcatFusion()

    img = torch.randn(2, 100, 64)
    meta = torch.randn(2, 5, 64)

    out = fusion(img, meta)

    assert out.shape == (2, 105, 64)


def test_concat_meta_first():
    """Test that metadata tokens come first."""
    fusion = ConcatFusion()

    img = torch.ones(1, 10, 16)
    meta = torch.zeros(1, 2, 16)

    out = fusion(img, meta)

    # First 2 tokens should be zeros (metadata)
    assert torch.all(out[0, :2] == 0)
    # Remaining tokens should be ones (image)
    assert torch.all(out[0, 2:] == 1)