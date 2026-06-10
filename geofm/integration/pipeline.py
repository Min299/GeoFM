"""geofm.integration.pipeline

GeoFM end-to-end pipeline.
"""
from __future__ import annotations


class GeoFMPipeline:
    """End-to-end pipeline for GeoFM inference."""

    def __init__(
        self,
        model,
        fusion=None,
        metadata_encoder=None,
    ):
        """Initialize pipeline.

        Args:
            model: Core GeoFM model
            fusion: Optional fusion module
            metadata_encoder: Optional metadata encoder
        """
        self.model = model

        self.fusion = fusion

        self.metadata_encoder = (
            metadata_encoder
        )

    def forward(
        self,
        image,
        metadata=None,
    ):
        """Run forward pass through pipeline.

        Args:
            image: Input image tensor
            metadata: Optional metadata dict

        Returns:
            Model output
        """
        return self.model(
            image
        )