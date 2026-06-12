"""geofm.models.backbones.terramind_wrapper

High-level wrapper around TerraMind backbone.
"""
from __future__ import annotations

from typing import Dict, Any, List

import torch
import torch.nn as nn

from geofm.models.backbones.terramind_backbone import (
    TerraMindBackbone,
)

from geofm.models.features.feature_extractor import (
    FeatureExtractor,
)


class TerraMindWrapper(nn.Module):
    """
    High-level wrapper around TerraMind.

    Responsibilities
    ----------------
    1. Receive modality dictionary
    2. Run TerraMind backbone
    3. Extract configured feature levels
    4. Return standardized feature list

    This is the canonical GeoFM interface to TerraMind.
    """

    def __init__(
        self,
        backbone: TerraMindBackbone,
        feature_indices=(2, 5, 8, 11),
    ):
        super().__init__()

        self.backbone = backbone

        self.extractor = FeatureExtractor(
            backbone=self.backbone,
            indices=feature_indices,
        )

    def forward(
        self,
        batch: Dict[str, Any],
    ) -> List[torch.Tensor]:

        return self.extractor(batch).to_list()

    def freeze_backbone(self):
        """Freeze the backbone parameters."""
        self.backbone.freeze()

    def unfreeze_backbone(self):
        """Unfreeze the backbone parameters."""
        self.backbone.unfreeze()

    def get_feature_info(self):
        """Get information about extracted features."""
        return self.backbone.get_feature_info()
