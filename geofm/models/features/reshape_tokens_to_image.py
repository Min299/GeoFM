"""geofm.models.features.reshape_tokens_to_image

Convert transformer tokens to 2D image format.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class ReshapeTokensToImage(nn.Module):
    """
    Convert transformer tokens:

        [B, N, D]

    into

        [B, D, H, W]

    Example:

        [8,196,768]
            ->
        [8,768,14,14]
    """

    def __init__(
        self,
        remove_cls_token: bool = False,
    ):
        super().__init__()

        self.remove_cls_token = (
            remove_cls_token
        )

    def forward(
        self,
        x: torch.Tensor,
    ):
        """Reshape tokens to image format.

        Args:
            x: Tensor of shape [B, N, D] where N = H*W tokens

        Returns:
            Tensor of shape [B, D, H, W]
        """
        if self.remove_cls_token:

            x = x[:, 1:, :]

        b, n, d = x.shape

        side = int(math.sqrt(n))

        if side * side != n:

            raise ValueError(
                f"Cannot reshape {n} tokens "
                f"into square feature map."
            )

        x = x.transpose(
            1,
            2,
        )

        x = x.reshape(
            b,
            d,
            side,
            side,
        )

        return x