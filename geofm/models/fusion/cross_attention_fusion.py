"""geofm.models.fusion.cross_attention_fusion

Cross-attention based fusion.
"""
from __future__ import annotations

import torch.nn as nn

from geofm.models.fusion.base_fusion import (
    BaseFusion,
)


class CrossAttentionFusion(
    BaseFusion
):
    """Cross-attention fusion between image and metadata tokens."""

    def __init__(
        self,
        embed_dim,
        num_heads=8,
    ):
        super().__init__()

        self.attn = (
            nn.MultiheadAttention(
                embed_dim,
                num_heads,
                batch_first=True,
            )
        )

    def forward(
        self,
        image_tokens,
        metadata_tokens,
    ):
        """Apply cross-attention from image to metadata tokens.

        Args:
            image_tokens: Tensor of shape (batch, seq_len, embed_dim)
            metadata_tokens: Tensor of shape (batch, meta_seq, embed_dim)

        Returns:
            Attended tensor of shape (batch, seq_len, embed_dim)
        """
        out, _ = self.attn(
            image_tokens,
            metadata_tokens,
            metadata_tokens,
        )

        return out