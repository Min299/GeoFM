"""geofm.research.fusion.film_fusion

FiLM (Feature-wise Linear Modulation) fusion.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from geofm.research.fusion.base_fusion import (
    BaseFusion,
)


class FiLMFusion(
    BaseFusion
):
    """Feature-wise Linear Modulation using metadata to generate gamma/beta."""

    def __init__(
        self,
        embed_dim,
    ):
        super().__init__()

        self.gamma = nn.Linear(
            embed_dim,
            embed_dim,
        )

        self.beta = nn.Linear(
            embed_dim,
            embed_dim,
        )

    def forward(
        self,
        image_tokens,
        metadata_tokens,
    ):
        """Apply FiLM conditioning to image tokens.

        Args:
            image_tokens: Tensor of shape (batch, seq_len, embed_dim)
            metadata_tokens: Tensor of shape (batch, meta_seq, embed_dim)

        Returns:
            Modulated tensor of shape (batch, seq_len, embed_dim)
        """
        meta = metadata_tokens.mean(
            dim=1
        )

        gamma = self.gamma(
            meta
        ).unsqueeze(1)

        beta = self.beta(
            meta
        ).unsqueeze(1)

        return (
            gamma
            * image_tokens
            + beta
        )