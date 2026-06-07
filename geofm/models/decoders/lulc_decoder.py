"""geofm.models.decoders.lulc_decoder

LULC segmentation decoder.

Wraps:
    - PyramidDecoder (FPN-style fusion)
    - SegmentationHead
"""
from __future__ import annotations

import torch.nn as nn

from geofm.models.decoders.pyramid_decoder import PyramidDecoder
from geofm.models.heads import SegmentationHead


class LULCDecoder(nn.Module):
    """LULC segmentation decoder."""

    def __init__(
        self,
        in_channels=(768, 768, 768, 768),
        decoder_channels=256,
        num_classes=10,
    ):
        super().__init__()

        self.pyramid = PyramidDecoder(
            in_channels=in_channels,
            decoder_channels=decoder_channels,
        )

        self.head = SegmentationHead(
            in_channels=decoder_channels,
            num_classes=num_classes,
        )

    def forward(self, features):
        """Forward pass.

        Args:
            features: List of feature maps [F2, F5, F8, F11]
                     Each shape: (B, C, H, W)

        Returns:
            Segmentation logits: (B, num_classes, H_out, W_out)
        """
        x = self.pyramid(features)
        return self.head(x)