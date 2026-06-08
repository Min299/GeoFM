"""tests/test_feature_extractor_adapter.py

Tests for FeatureExtractorAdapter.
"""
import pytest
import torch


class DummyBackbone:
    """Dummy backbone for testing."""

    def extract_features(self, batch):
        return [
            torch.randn(2, 768, 64, 64) for _ in range(12)
        ]


class DummyBackboneCallable:
    """Dummy callable backbone for testing."""

    def __call__(self, batch):
        return [
            torch.randn(2, 768, 64, 64) for _ in range(12)
        ]


class TestFeatureExtractorAdapter:
    """Test FeatureExtractorAdapter class."""

    def test_extract_with_extract_features(self):
        """Should extract features from backbone with extract_features."""
        from geofm.integration import FeatureExtractorAdapter

        backbone = DummyBackbone()
        extractor = FeatureExtractorAdapter(backbone)

        features = extractor.extract({})

        assert len(features) == 4
        assert all(f.shape[1] == 768 for f in features)

    def test_extract_with_call(self):
        """Should extract features from callable backbone."""
        from geofm.integration import FeatureExtractorAdapter

        backbone = DummyBackboneCallable()
        extractor = FeatureExtractorAdapter(backbone)

        features = extractor.extract({})

        assert len(features) == 4

    def test_custom_indices(self):
        """Should use custom feature indices."""
        from geofm.integration import FeatureExtractorAdapter

        backbone = DummyBackbone()
        extractor = FeatureExtractorAdapter(backbone, feature_indices=[0, 3, 6, 9])

        features = extractor.extract({})

        assert len(features) == 4

    def test_set_indices(self):
        """Should allow setting new indices."""
        from geofm.integration import FeatureExtractorAdapter

        backbone = DummyBackbone()
        extractor = FeatureExtractorAdapter(backbone)

        extractor.set_indices([1, 4, 7, 10])

        assert extractor.get_indices() == [1, 4, 7, 10]

    def test_get_indices(self):
        """Should return current indices."""
        from geofm.integration import FeatureExtractorAdapter

        backbone = DummyBackbone()
        extractor = FeatureExtractorAdapter(backbone)

        assert extractor.get_indices() == [2, 5, 8, 11]

    def test_default_indices(self):
        """Default indices should be [2, 5, 8, 11]."""
        from geofm.integration import FeatureExtractorAdapter

        backbone = DummyBackbone()
        extractor = FeatureExtractorAdapter(backbone)

        assert extractor.feature_indices == [2, 5, 8, 11]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])