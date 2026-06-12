"""geofm.models.metadata.metadata_tokenizer

Converts metadata to token format for fusion.
"""
from __future__ import annotations

import torch

from geofm.models.metadata.metadata_projector import (
    MetadataProjector,
)


class MetadataTokenizer:
    """Converts metadata vector to token representation."""

    def __init__(
        self,
        metadata_dim,
        embed_dim,
    ):
        """Initialize tokenizer with projector.

        Args:
            metadata_dim: Dimension of raw metadata vector
            embed_dim: Target embedding dimension
        """
        self.projector = (
            MetadataProjector(
                metadata_dim,
                embed_dim,
            )
        )

    def __call__(
        self,
        metadata,
    ):
        """Tokenize metadata.

        Args:
            metadata: Tensor of shape (batch, metadata_dim)

        Returns:
            Tensor of shape (batch, 1, embed_dim) - single token per sample
        """
        token = (
            self.projector(
                metadata
            )
        )

        return token.unsqueeze(
            1
        )