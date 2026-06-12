"""geofm.models.metadata.metadata_projector

Projects metadata vectors to embedding dimension.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MetadataProjector(nn.Module):
    """Linear projection from metadata dimension to embedding dimension."""

    def __init__(
        self,
        metadata_dim: int,
        embed_dim: int,
    ):
        super().__init__()

        self.proj = nn.Linear(
            metadata_dim,
            embed_dim,
        )

    def forward(
        self,
        metadata,
    ):
        """Project metadata vector.

        Args:
            metadata: Tensor of shape (batch, metadata_dim)

        Returns:
            Tensor of shape (batch, embed_dim)
        """
        return self.proj(
            metadata
        )