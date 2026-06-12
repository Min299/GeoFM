"""tests/test_real_forward.py

Tests for real forward pass through model.
"""
import pytest
import torch


class DummyBackbone:
    """Dummy backbone that returns 4 feature levels."""

    def __call__(self, batch):
        """Return 4 features at different resolutions."""
        return [
            torch.randn(2, 768, 64, 64),
            torch.randn(2, 768, 32, 32),
            torch.randn(2, 768, 16, 16),
            torch.randn(2, 768, 8, 8),
        ]


class TestRealForward:
    """Test real forward passes."""

    def test_backbone_forward(self):
        """Backbone should return 4 features."""
        backbone = DummyBackbone()

        feats = backbone({})

        assert len(feats) == 4

    def test_backbone_feature_shapes(self):
        """Features should have correct shapes."""
        backbone = DummyBackbone()

        feats = backbone({})

        expected_shapes = [
            (2, 768, 64, 64),
            (2, 768, 32, 32),
            (2, 768, 16, 16),
            (2, 768, 8, 8),
        ]

        for feat, expected in zip(feats, expected_shapes):
            assert feat.shape == expected

    def test_feature_inspector(self):
        """FeatureInspector should work."""
        from geofm.debug.feature_inspector import FeatureInspector

        backbone = DummyBackbone()
        feats = backbone({})

        # Should not raise
        FeatureInspector.inspect_feature_list(feats)

    def test_forward_trace(self):
        """ForwardTrace should work."""
        from geofm.debug.forward_trace import ForwardTrace

        backbone = DummyBackbone()
        feats = backbone({})

        # Should not raise
        ForwardTrace.print_features(feats)

    def test_feature_inspector_summary(self):
        """FeatureInspector.summary should work."""
        from geofm.debug.feature_inspector import FeatureInspector

        backbone = DummyBackbone()
        feats = backbone({})

        summary = FeatureInspector.summary(feats)

        assert summary["count"] == 4
        assert len(summary["shapes"]) == 4
        assert summary["all_same_channels"] is True
        assert summary["channel_dim"] == 768


if __name__ == "__main__":
    pytest.main([__file__, "-v"])