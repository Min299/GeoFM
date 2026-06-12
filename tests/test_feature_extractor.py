"""tests/test_feature_extractor.py

Tests for FeatureExtractor.
"""
import torch

from geofm.models.features.feature_extractor import (
    FeatureExtractor,
)


class DummyBackbone:

    def __call__(
        self,
        batch,
    ):
        return [
            torch.randn(
                2,
                196,
                768,
            )
            for _ in range(12)
        ]


def test_extractor():
    """Test FeatureExtractor returns correct number of features."""
    extractor = (
        FeatureExtractor(
            DummyBackbone()
        )
    )

    features = extractor(
        {}
    )

    assert len(
        features.to_list()
    ) == 4


def test_extractor_custom_indices():
    """Test FeatureExtractor with custom indices."""
    extractor = (
        FeatureExtractor(
            DummyBackbone(),
            indices=[2, 5],
        )
    )

    features = extractor({})

    assert len(
        features.to_list()
    ) == 2