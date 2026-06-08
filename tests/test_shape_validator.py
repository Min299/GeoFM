"""tests/test_shape_validator.py

Tests for ShapeValidator.
"""
import pytest
import torch


class TestShapeValidator:
    """Test ShapeValidator class."""

    def test_validate_feature_list_success(self):
        """Valid feature list should pass."""
        from geofm.integration import ShapeValidator

        features = [
            torch.randn(2, 768, 64, 64),
            torch.randn(2, 768, 32, 32),
            torch.randn(2, 768, 16, 16),
            torch.randn(2, 768, 8, 8),
        ]

        # Should not raise
        ShapeValidator.validate_feature_list(features)

    def test_validate_feature_list_wrong_count(self):
        """Wrong number of features should raise."""
        from geofm.integration import ShapeValidator

        features = [
            torch.randn(2, 768, 64, 64),
            torch.randn(2, 768, 32, 32),
        ]

        with pytest.raises(ValueError, match="Expected 4 features"):
            ShapeValidator.validate_feature_list(features)

    def test_validate_feature_list_wrong_dims(self):
        """Non-4D features should raise."""
        from geofm.integration import ShapeValidator

        features = [
            torch.randn(2, 768, 64),  # 3D instead of 4D
            torch.randn(2, 768, 32, 32),
            torch.randn(2, 768, 16, 16),
            torch.randn(2, 768, 8, 8),
        ]

        with pytest.raises(ValueError, match="must be 4D"):
            ShapeValidator.validate_feature_list(features)

    def test_validate_batch_size_success(self):
        """Same batch sizes should pass."""
        from geofm.integration import ShapeValidator

        features = [
            torch.randn(2, 768, 64, 64),
            torch.randn(2, 768, 32, 32),
            torch.randn(2, 768, 16, 16),
            torch.randn(2, 768, 8, 8),
        ]

        ShapeValidator.validate_batch_size(features)

    def test_validate_batch_size_mismatch(self):
        """Different batch sizes should raise."""
        from geofm.integration import ShapeValidator

        features = [
            torch.randn(2, 768, 64, 64),
            torch.randn(3, 768, 32, 32),  # Different batch size
            torch.randn(2, 768, 16, 16),
            torch.randn(2, 768, 8, 8),
        ]

        with pytest.raises(ValueError, match="batch size"):
            ShapeValidator.validate_batch_size(features)

    def test_validate_channels_success(self):
        """Same channels should pass."""
        from geofm.integration import ShapeValidator

        features = [
            torch.randn(2, 768, 64, 64),
            torch.randn(2, 768, 32, 32),
            torch.randn(2, 768, 16, 16),
            torch.randn(2, 768, 8, 8),
        ]

        ShapeValidator.validate_channels(features)

    def test_validate_channels_mismatch(self):
        """Different channels should raise."""
        from geofm.integration import ShapeValidator

        features = [
            torch.randn(2, 768, 64, 64),
            torch.randn(2, 512, 32, 32),  # Different channels
            torch.randn(2, 768, 16, 16),
            torch.randn(2, 768, 8, 8),
        ]

        with pytest.raises(ValueError, match="channels"):
            ShapeValidator.validate_channels(features)

    def test_validate_all(self):
        """validate_all should run all checks."""
        from geofm.integration import ShapeValidator

        features = [
            torch.randn(2, 768, 64, 64),
            torch.randn(2, 768, 32, 32),
            torch.randn(2, 768, 16, 16),
            torch.randn(2, 768, 8, 8),
        ]

        ShapeValidator.validate_all(features)

    def test_get_feature_info(self):
        """Should return feature info."""
        from geofm.integration import ShapeValidator

        features = [
            torch.randn(2, 768, 64, 64),
            torch.randn(2, 768, 32, 32),
            torch.randn(2, 768, 16, 16),
            torch.randn(2, 768, 8, 8),
        ]

        info = ShapeValidator.get_feature_info(features)

        assert info["num_features"] == 4
        assert info["batch_size"] == 2
        assert info["channels"] == 768
        assert len(info["feature_shapes"]) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])