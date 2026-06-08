"""geofm.debug.feature_inspector

Feature inspection utilities for debugging feature extraction.
"""
from __future__ import annotations

from typing import List, Optional

import torch


class FeatureInspector:
    """Inspect feature tensors during forward pass.

    Usage:
        features = model.extract_features(batch)
        FeatureInspector.inspect(features, title="After Backbone")
    """

    @staticmethod
    def inspect(
        features: List[torch.Tensor],
        title: str = "Features",
    ) -> None:
        """Inspect a list of feature tensors.

        Args:
            features: List of feature tensors
            title: Title for the inspection output
        """
        print("\n" + "=" * 60)
        print(title)
        print("=" * 60)

        for idx, feat in enumerate(features):
            print(
                f"F{idx}: "
                f"shape={tuple(feat.shape)} "
                f"dtype={feat.dtype} "
                f"device={feat.device}"
            )

        print("=" * 60)

    @staticmethod
    def inspect_feature_list(
        features: List[torch.Tensor],
    ) -> None:
        """Validate and inspect a feature list.

        Args:
            features: List of feature tensors to validate

        Raises:
            AssertionError: If features don't meet requirements
        """
        assert isinstance(features, (list, tuple)), \
            f"Features must be list or tuple, got {type(features)}"

        assert len(features) == 4, \
            f"Expected 4 features, got {len(features)}"

        for idx, feat in enumerate(features):
            assert torch.is_tensor(feat), \
                f"F{idx} is not a tensor, got {type(feat)}"

    @staticmethod
    def compare_features(
        features_a: List[torch.Tensor],
        features_b: List[torch.Tensor],
        title_a: str = "Features A",
        title_b: str = "Features B",
    ) -> None:
        """Compare two feature lists.

        Args:
            features_a: First feature list
            features_b: Second feature list
            title_a: Title for first list
            title_b: Title for second list
        """
        print("\n" + "=" * 60)
        print("Feature Comparison")
        print("=" * 60)

        FeatureInspector.inspect(features_a, title_a)
        FeatureInspector.inspect(features_b, title_b)

        # Check shapes match
        for idx, (fa, fb) in enumerate(zip(features_a, features_b)):
            match = "✓" if fa.shape == fb.shape else "✗"
            print(f"F{idx}: {match} {tuple(fa.shape)} vs {tuple(fb.shape)}")

        print("=" * 60)

    @staticmethod
    def check_gradients(
        features: List[torch.Tensor],
        name: str = "Features",
    ) -> bool:
        """Check if features have gradients.

        Args:
            features: List of feature tensors
            name: Name for the check

        Returns:
            True if any feature has gradients
        """
        has_grad = False

        for idx, feat in enumerate(features):
            if feat.requires_grad:
                print(f"{name} F{idx}: requires_grad=True")
                has_grad = True
            else:
                print(f"{name} F{idx}: requires_grad=False")

        return has_grad

    @staticmethod
    def summary(features: List[torch.Tensor]) -> dict:
        """Get a summary dictionary of feature statistics.

        Args:
            features: List of feature tensors

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            "count": len(features),
            "shapes": [tuple(f.shape) for f in features],
            "dtypes": [str(f.dtype) for f in features],
            "devices": [str(f.device) for f in features],
        }

        # Check if all have same channel dimension
        channels = [f.shape[1] if len(f.shape) >= 2 else None for f in features]
        summary["all_same_channels"] = len(set(channels)) == 1 if all(c is not None for c in channels) else None
        summary["channel_dim"] = channels[0] if all(c == channels[0] for c in channels) else channels

        return summary