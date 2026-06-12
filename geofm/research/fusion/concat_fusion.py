"""geofm.research.fusion.concat_fusion

Concatenation-based fusion.
"""
from __future__ import annotations

import torch

from geofm.research.fusion.base_fusion import (
    BaseFusion,
)


class ConcatFusion(
    BaseFusion
):
    """Concatenates metadata tokens before image tokens."""

    def forward(
        self,
        image_tokens,
        metadata_tokens,
    ):
        """Concatenate metadata tokens with image tokens.

        Args:
            image_tokens: Tensor of shape (batch, seq_len, embed_dim)
            metadata_tokens: Tensor of shape (batch, meta_seq, embed_dim)

        Returns:
            Concatenated tensor of shape (batch, seq_len + meta_seq, embed_dim)
        """
        return torch.cat(
            [
                metadata_tokens,
                image_tokens,
            ],
            dim=1,
        )