"""geofm.models.features.learned_interpolate_to_pyramid

Learned interpolation to pyramidal feature maps.
"""
from __future__ import annotations

import torch.nn as nn


class LearnedInterpolateToPyramid(
    nn.Module
):
    """Learned interpolation to pyramidal feature maps.

    Projects features to a unified channel dimension using
    learned 1x1 convolutions per level.
    """

    def __init__(
        self,
        out_channels=256,
    ):
        super().__init__()

        self.out_channels = (
            out_channels
        )

        # Placeholder for projections - built lazily per forward pass
        self._projection_cache = {}

    def _build_projection(
        self,
        in_channels,
    ):
        """Build or retrieve projection for given channel count.

        Args:
            in_channels: Input channel dimension

        Returns:
            nn.Conv2d projection layer
        """
        key = in_channels

        if key not in self._projection_cache:
            self._projection_cache[key] = nn.Conv2d(
                in_channels,
                self.out_channels,
                kernel_size=1,
            ).to(self._get_device())

        return self._projection_cache[key]

    def _get_device(self):
        """Get device from first parameter or default to cpu."""
        for module in self._projection_cache.values():
            return next(module.parameters()).device
        return nn.Parameter().device if hasattr(nn.Parameter(), 'device') else 'cpu'

    def forward(
        self,
        features,
    ):
        """Project features to pyramidal format.

        Args:
            features: List of feature tensors [B, D, H, W]

        Returns:
            List of projected tensors, all with out_channels
        """
        outputs = []

        for feat in features:

            c = feat.shape[1]

            proj = self._build_projection(c)

            feat = proj(feat)

            outputs.append(
                feat
            )

        return outputs