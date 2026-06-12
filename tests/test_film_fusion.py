"""tests/test_film_fusion.py

Tests for FiLMFusion.
"""
import torch

from geofm.models.fusion.film_fusion import (
    FiLMFusion,
)


def test_film():
    """Test FiLM fusion."""
    fusion = FiLMFusion(
        128
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


def test_film_preserves_shape():
    """Test that FiLM preserves spatial dimensions."""
    fusion = FiLMFusion(64)

    img = torch.randn(4, 50, 64)
    meta = torch.randn(4, 3, 64)

    out = fusion(img, meta)

    assert out.shape == img.shape


def test_film_modulation():
    """Test that FiLM actually modulates the image tokens."""
    fusion = FiLMFusion(32)

    img = torch.ones(1, 10, 32)
    meta = torch.ones(1, 1, 32) * 2  # metadata influences gamma/beta

    out = fusion(img, meta)

    # Output should be different from input due to modulation
    assert not torch.allclose(out, img)


def test_film_trainable():
    """Test that FiLM has trainable parameters."""
    fusion = FiLMFusion(64)

    params = list(fusion.parameters())
    assert len(params) == 4  # gamma_w, gamma_b, beta_w, beta_b