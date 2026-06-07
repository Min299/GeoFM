from __future__ import annotations

import torch
import torch.nn as nn


class FeatureAdapter(nn.Module):
    """
    Bottleneck feature adapter.

    x -> Down -> GELU -> Up -> Residual

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

        self.down = nn.Linear(
            dim,
            bottleneck_dim,
            bias=False,
        )

        self.act = nn.GELU()

        self.dropout = nn.Dropout(
            dropout
        )

        self.up = nn.Linear(
            bottleneck_dim,
            dim,
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