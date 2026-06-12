"""tests/test_cross_attention_fusion.py

Tests for CrossAttentionFusion.
"""
import torch

from geofm.models.fusion.cross_attention_fusion import (
    CrossAttentionFusion,
)


def test_cross_attention():
    """Test cross-attention fusion."""
    fusion = (
        CrossAttentionFusion(
            128
        )
    )

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

    assert out.shape == (
        2,
        196,
        128,
    )


def test_cross_attention_preserves_shape():
    """Test that cross-attention preserves spatial dimensions."""
    fusion = CrossAttentionFusion(64)

    img = torch.randn(4, 50, 64)
    meta = torch.randn(4, 5, 64)

    out = fusion(img, meta)

    assert out.shape == img.shape


def test_cross_attention_num_heads():
    """Test with different number of heads."""
    fusion = CrossAttentionFusion(64, num_heads=4)

    img = torch.randn(2, 100, 64)
    meta = torch.randn(2, 3, 64)

    out = fusion(img, meta)

    assert out.shape == (2, 100, 64)


def test_cross_attention_trainable():
    """Test that cross-attention has trainable parameters."""
    fusion = CrossAttentionFusion(64)

    params = list(fusion.parameters())
    assert len(params) > 0
    assert all(p.requires_grad for p in params)