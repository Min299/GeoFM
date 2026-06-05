"""geofm.models.backbones.feature_extractor

Standardize feature extraction for:
- Segmentation
- Distillation
- Shared Training

Provides unified interface to extract intermediate features
at specific indices [2, 5, 8, 11] from TerraMind backbone.
"""
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    """Extract intermediate features from TerraMind backbone.

    Standardizes feature extraction across all use cases:
    - Segmentation decoder input
    - Distillation loss computation
    - Multi-task training

    Usage:
        extractor = FeatureExtractor(backbone, feature_indices=(2, 5, 8, 11))
        features = extractor(batch)  # Returns list of feature tensors

    Attributes:
        backbone: The underlying model (TerraMind or compatible)
        feature_indices: Which transformer layers to extract
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_indices: Tuple[int, ...] = (2, 5, 8, 11)
    ):
        """Initialize feature extractor.

        Args:
            backbone: Model to extract features from
            feature_indices: Tuple of layer indices to extract
                Default: (2, 5, 8, 11) for base/tiny/small models
                For large models use: (5, 11, 17, 23)
        """
        super().__init__()
        self.backbone = backbone
        self.feature_indices = feature_indices

    def forward(
        self,
        batch: Dict[str, Any]
    ) -> List[torch.Tensor]:
        """Extract features from batch.

        Args:
            batch: Input dict with modalities or raw tensor

        Returns:
            List of feature tensors at specified indices
        """
        outputs = self.backbone(batch)

        # Handle different output formats
        if isinstance(outputs, list):
            # Already a list of features per layer
            features = outputs
        elif isinstance(outputs, dict) and "features" in outputs:
            # Dict with features key
            features = outputs["features"]
        else:
            # Assume single tensor - repeat for each index
            features = [outputs] * len(self.feature_indices)

        # Extract specific indices
        extracted = []
        for idx in self.feature_indices:
            if idx < len(features):
                extracted.append(features[idx])
            else:
                # Fallback: use last available
                extracted.append(features[-1])

        return extracted

    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about available features."""
        return {
            "feature_indices": self.feature_indices,
            "num_features": len(self.feature_indices),
        }


def create_feature_extractor(
    backbone: nn.Module,
    model_name: str = "terramind_v1_base"
) -> FeatureExtractor:
    """Create a feature extractor for the given backbone.

    Args:
        backbone: Model to extract features from
        model_name: Model variant to determine correct indices

    Returns:
        Configured FeatureExtractor
    """
    # Default indices by model variant
    if "large" in model_name:
        indices = (5, 11, 17, 23)
    else:
        indices = (2, 5, 8, 11)

    return FeatureExtractor(backbone, feature_indices=indices)