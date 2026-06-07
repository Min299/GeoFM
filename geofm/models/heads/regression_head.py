from __future__ import annotations


import torch
import torch.nn as nn




class RegressionHead(nn.Module):
    """
    Pixel-wise regression.


    Example:
        NDVI
    """


    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
    ):
        super().__init__()


        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
        )


    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:


        return self.conv(x)