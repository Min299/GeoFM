"""geofm.models.tasks.segmentation_model

Segmentation model using TerraMind backbone and decoder.
"""
from __future__ import annotations

import torch.nn as nn

from geofm.models.backbones.terramind_wrapper import (
    TerraMindWrapper,
)

from geofm.models.features.reshape_tokens_to_image import (
    ReshapeTokensToImage,
)


class SegmentationModel(
    nn.Module
):
    """Segmentation model with TerraMind backbone.

    Architecture:
        Input Batch
            ->
        TerraMind Backbone
            ->
        Token -> Image Reshape
            ->
        Decoder
            ->
        Segmentation Output
    """

    def __init__(
        self,
        backbone,
        decoder,
    ):
        super().__init__()

        self.backbone = (
            backbone
        )

        self.decoder = (
            decoder
        )

        self.reshape = (
            ReshapeTokensToImage()
        )

    def forward(
        self,
        batch,
    ):
        """Forward pass for segmentation.

        Args:
            batch: Input batch dictionary

        Returns:
            Segmentation output
        """
        features = (
            self.backbone(
                batch
            )
        )

        image_features = [

            self.reshape(f)

            for f in features
        ]

        return self.decoder(
            image_features
        )