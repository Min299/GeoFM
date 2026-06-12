"""tests/test_token_to_image.py

Tests for ReshapeTokensToImage.
"""
import torch

from geofm.models.features.reshape_tokens_to_image import (
    ReshapeTokensToImage,
)


def test_token_to_image():
    """Test reshape from [B, N, D] to [B, D, H, W]."""
    layer = (
        ReshapeTokensToImage()
    )

    x = torch.randn(
        2,
        196,
        768,
    )

    y = layer(x)

    assert y.shape == (
        2,
        768,
        14,
        14,
    )


def test_token_to_image_with_cls():
    """Test reshape with cls token removal."""
    layer = (
        ReshapeTokensToImage(
            remove_cls_token=True
        )
    )

    x = torch.randn(
        2,
        197,
        768,
    )

    y = layer(x)

    assert y.shape == (
        2,
        768,
        14,
        14,
    )