from __future__ import annotations

import torch
import torch.nn as nn


class FeatureAdapter(nn.Module):
    """
    Bottleneck feature adapter for spatial features.

    Operates on (B, C, H, W) tensors.
    
    x -> ChannelProj(C→bottleneck) -> GELU -> ChannelProj(bottleneck→C) -> + residual

    Output:
        y = x + Adapter(x)
    """

    def __init__(
        self,
        dim: int,
        bottleneck_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Project channels: C -> bottleneck_dim
        self.down = nn.Conv2d(
            dim,
            bottleneck_dim,
            kernel_size=1,
            bias=False,
        )

        self.act = nn.GELU()

        self.dropout = nn.Dropout(
            dropout
        )

        # Project channels: bottleneck_dim -> C
        self.up = nn.Conv2d(
            bottleneck_dim,
            dim,
            kernel_size=1,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        residual = x

        x = self.down(x)

        x = self.act(x)

        x = self.dropout(x)

        x = self.up(x)

        return residual + x