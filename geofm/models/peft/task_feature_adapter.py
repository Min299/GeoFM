from __future__ import annotations

import torch.nn as nn

from .feature_adapter import (
    FeatureAdapter
)


class TaskFeatureAdapter(
    nn.Module
):
    """
    One adapter per feature level.

    F2
    F5
    F8
    F11
    """

    def __init__(
        self,
        feature_dims,
        bottleneck_dim=64,
    ):
        super().__init__()

        self.adapters = nn.ModuleList(
            [
                FeatureAdapter(
                    dim=d,
                    bottleneck_dim=bottleneck_dim,
                )
                for d in feature_dims
            ]
        )

    def forward(
        self,
        features,
    ):

        return [
            adapter(feature)
            for adapter, feature
            in zip(
                self.adapters,
                features,
            )
        ]