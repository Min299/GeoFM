"""geofm.integration.shape_validator

Validators for feature map shapes.
"""
from __future__ import annotations

from typing import List, Any


class ShapeValidator:
    """Validates feature map shapes for GeoFM compatibility.

    Ensures all feature maps conform to expected shapes:
    - 4 features required (F2, F5, F8, F11)
    - All features must be 4D tensors
    - All features must have same batch size
    - All features must have same channel dimension
    """

    @staticmethod
    def validate_feature_list(features: List[Any]) -> None:
        """Validate feature list has correct structure.

        Args:
            features: List of feature tensors

        Raises:
            ValueError: If feature list is invalid
        """
        if len(features) != 4:
            raise ValueError(
                f"Expected 4 features [F2, F5, F8, F11], got {len(features)}"
            )

        for idx, feature in enumerate(features):
            if feature.ndim != 4:
                raise ValueError(
                    f"Feature {idx} must be 4D tensor (B, C, H, W), "
                    f"got {feature.ndim}D"
                )

    @staticmethod
    def validate_batch_size(features: List[Any]) -> None:
        """Validate all features have same batch size.

        Args:
            features: List of feature tensors

        Raises:
            ValueError: If batch sizes don't match
        """
        if not features:
            return

        batch_size = features[0].shape[0]

        for idx, feature in enumerate(features):
            if feature.shape[0] != batch_size:
                raise ValueError(
                    f"Feature {idx} has batch size {feature.shape[0]}, "
                    f"expected {batch_size}"
                )

    @staticmethod
    def validate_channels(features: List[Any]) -> None:
        """Validate all features have same channel dimension.

        Args:
            features: List of feature tensors

        Raises:
            ValueError: If channel dimensions don't match
        """
        if not features:
            return

        channels = features[0].shape[1]

        for idx, feature in enumerate(features):
            if feature.shape[1] != channels:
                raise ValueError(
                    f"Feature {idx} has {feature.shape[1]} channels, "
                    f"expected {channels}"
                )

    @staticmethod
    def validate_all(features: List[Any]) -> None:
        """Run all validations.

        Args:
            features: List of feature tensors

        Raises:
            ValueError: If any validation fails
        """
        ShapeValidator.validate_feature_list(features)
        ShapeValidator.validate_batch_size(features)
        ShapeValidator.validate_channels(features)

    @staticmethod
    def get_feature_info(features: List[Any]) -> dict:
        """Get information about feature shapes.

        Args:
            features: List of feature tensors

        Returns:
            Dictionary with shape information
        """
        if not features:
            return {"num_features": 0}

        return {
            "num_features": len(features),
            "batch_size": features[0].shape[0],
            "channels": features[0].shape[1],
            "feature_shapes": [(f.shape[2], f.shape[3]) for f in features],
            "stride_pattern": [
                features[0].shape[2] // f.shape[2] if f.shape[2] > 0 else 0
                for f in features
            ],
        }