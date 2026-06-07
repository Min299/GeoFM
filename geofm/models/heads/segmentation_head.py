from __future__ import annotations


import torch
import torch.nn as nn




class SegmentationHead(nn.Module):
    """
    Final segmentation logits layer.


    Input:
        (B, C, H, W)


    Output:
        (B, num_classes, H, W)
    """


    def __init__(
        self,
        in_channels: int,
        num_classes: int,
    ):
        super().__init__()


        self.conv = nn.Conv2d(
            in_channels,
            num_classes,
            kernel_size=1,
        )


    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:


        return self.conv(x)