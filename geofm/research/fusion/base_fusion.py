"""geofm.research.fusion.base_fusion

Base class for fusion modules.
"""
from __future__ import annotations

import torch.nn as nn


class BaseFusion(
    nn.Module
):
    """Base class for fusion modules."""

    def forward(
        self,
        image_tokens,
        metadata_tokens,
    ):
        """Fuse image tokens with metadata tokens.

        Args:
            image_tokens: Tensor of shape (batch, seq_len, embed_dim)
            metadata_tokens: Tensor of shape (batch, meta_seq, embed_dim)

        Returns:
            Fused tensor
        """
        raise NotImplementedError