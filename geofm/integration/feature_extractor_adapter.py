"""geofm.integration.feature_extractor_adapter

Wrapper for extracting specific features from backbones.
"""
from __future__ import annotations

from typing import List, Any, Optional


class FeatureExtractorAdapter:
    """Adapter for extracting specific features from a backbone.

    Wraps a backbone and extracts features at specified layer indices.

    Usage:
        extractor = FeatureExtractorAdapter(
            backbone,
            feature_indices=[2, 5, 8, 11]
        )
        features = extractor.extract(batch)
        # Returns [F2, F5, F8, F11]
    """

    def __init__(
        self,
        backbone: Any,
        feature_indices: Optional[List[int]] = None,
    ):
        """Initialize feature extractor.

        Args:
            backbone: The backbone model to extract features from
            feature_indices: Indices of layers to extract features from
                             Defaults to [2, 5, 8, 11]
        """
        self.backbone = backbone
        self.feature_indices = feature_indices if feature_indices is not None else [2, 5, 8, 11]

    def extract(self, batch: dict) -> List[Any]:
        """Extract features from batch.

        Args:
            batch: Input batch dictionary

        Returns:
            List of feature maps at specified indices
        """
        if hasattr(self.backbone, 'extract_features'):
            features = self.backbone.extract_features(batch)
        elif hasattr(self.backbone, '__call__'):
            features = self.backbone(batch)
        else:
            raise ValueError("Backbone must have extract_features or __call__ method")

        return [features[i] for i in self.feature_indices]

    def set_indices(self, indices: List[int]) -> None:
        """Set feature extraction indices.

        Args:
            indices: New layer indices
        """
        self.feature_indices = indices

    def get_indices(self) -> List[int]:
        """Get current feature indices.

        Returns:
            List of current indices
        """
        return self.feature_indices