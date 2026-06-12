"""geofm.integration.backbone_adapter

Unified interface for all GeoFM backbones.
All backbones (TerraMind, Prithvi, StudentViT) must implement this.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Any


class BackboneAdapter(ABC):
    """Abstract base class for backbone adapters.

    All GeoFM backbones must implement this interface to ensure
    compatibility with the adapter and decoder banks.

    Usage:
        class TerraMindAdapter(BackboneAdapter):
            def get_features(self, batch):
                return self.backbone.extract_features(batch)

        adapter = TerraMindAdapter(backbone)
        features = adapter.get_features(batch)
    """

    @abstractmethod
    def get_features(self, batch: dict) -> List[Any]:
        """Extract features from input batch.

        Returns:
            List of 4 feature maps [F2, F5, F8, F11]
            Each feature should be a 4D tensor (B, C, H, W)
        """
        raise NotImplementedError

    @abstractmethod
    def feature_dim(self) -> int:
        """Get the channel dimension of features.

        Returns:
            Number of channels per feature map
        """
        raise NotImplementedError

    @abstractmethod
    def num_layers(self) -> int:
        """Get the number of transformer layers.

        Returns:
            Number of layers in the backbone
        """
        raise NotImplementedError

    @abstractmethod
    def freeze(self) -> None:
        """Freeze backbone parameters for efficient fine-tuning."""
        raise NotImplementedError

    @abstractmethod
    def unfreeze(self) -> None:
        """Unfreeze backbone parameters for full fine-tuning."""
        raise NotImplementedError

    def get_layer_indices(self) -> List[int]:
        """Get the layer indices for feature extraction.

        Default implementation returns [2, 5, 8, 11]
        which corresponds to 25%, 50%, 75%, and last layer.

        Returns:
            List of layer indices to extract features from
        """
        return [2, 5, 8, 11]

    @property
    def is_frozen(self) -> bool:
        """Check if backbone is frozen.

        Returns:
            True if all parameters require no gradients
        """
        return not any(p.requires_grad for p in self.parameters())

    def parameters(self):
        """Get backbone parameters.

        Returns:
            Iterator of parameters
        """
        return iter([])  # Override in subclass