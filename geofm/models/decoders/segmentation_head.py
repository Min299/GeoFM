"""geofm.models.decoders.segmentation_head

Unified decoder interface for segmentation.
Not a decoder itself - wraps the decoder with a standard interface.
"""
from typing import List, Optional

import torch
import torch.nn as nn


class SegmentationHead(nn.Module):
    """Wrapper for segmentation decoders.

    Provides a unified interface for different decoder types:
    - UNetDecoder
    - DeepLabV3Decoder
    - etc.

    Usage:
        head = SegmentationHead(decoder)
        output = head(features)  # List[Tensor] -> Segmentation logits
    """

    def __init__(
        self,
        decoder: nn.Module,
        num_classes: int = 2
    ):
        """Initialize segmentation head.

        Args:
            decoder: The decoder module (e.g., UNetDecoder)
            num_classes: Number of output classes
        """
        super().__init__()
        self.decoder = decoder
        self.num_classes = num_classes

    def forward(
        self,
        features: List[torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass through decoder.

        Args:
            features: List of feature tensors from backbone
                Typically from feature indices [2, 5, 8, 11]

        Returns:
            Segmentation logits with shape (B, num_classes, H, W)
        """
        return self.decoder(features)

    def get_decoder_info(self) -> dict:
        """Get information about the decoder."""
        return {
            "num_classes": self.num_classes,
            "decoder_type": type(self.decoder).__name__,
        }