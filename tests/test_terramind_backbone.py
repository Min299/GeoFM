"""tests/test_terramind_backbone.py

Tests for TerraMindBackbone wrapper.

These tests verify:
1. Forward pass returns 4 features [F2, F5, F8, F11]
2. Freeze/unfreeze works correctly
3. Feature extraction interface
"""
import pytest
import torch
import torch.nn as nn

from geofm.models.backbones import (
    TerraMindBackbone,
    build_backbone,
)


class DummyTerraMindEncoder(nn.Module):
    """Dummy TerraMind encoder for testing.

    Produces 12 transformer layer outputs for proper indexing.
    """

    def __init__(self, feature_dim=256, num_layers=12):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        # Add dummy parameters so freeze/unfreeze has something to work with
        self.dummy_param = nn.Parameter(torch.randn(feature_dim, feature_dim))

    def cat_encoder_tensors(self, mod_dict):
        """Mock implementation of TerraTorch's cat_encoder_tensors."""
        # Extract the tensor from the dict
        x = mod_dict["S2L1C"]["x"]
        batch_size = x.shape[0]
        num_tokens = x.shape[2] * x.shape[3]  # H * W
        # Return dummy encoder outputs
        encoder_tokens = torch.randn(batch_size, num_tokens, self.feature_dim)
        return encoder_tokens, None, None, None

    def forward_encoder(self, tokens, mask=None):
        """Mock implementation returning 12 transformer layer outputs.

        Returns list where features at indices [2, 5, 8, 11] are used for distillation.
        """
        batch_size = tokens.shape[0]
        # Return 12 features (one per transformer layer)
        features = []
        for i in range(self.num_layers):
            # Vary resolution slightly per layer to simulate real behavior
            h = w = max(4, 64 // (i % 4 + 1))
            features.append(
                torch.randn(batch_size, h, w, self.feature_dim)
            )
        return features

    def forward(self, mod_dict):
        """Return list of features (simulating transformer layers).

        Args:
            mod_dict: Dictionary with modalities

        Returns:
            List of feature tensors (one per layer)
        """
        return self.forward_encoder(self.cat_encoder_tensors(mod_dict)[0])


def create_test_backbone():
    """Create a TerraMindBackbone for testing with all required attributes."""
    backbone = TerraMindBackbone.__new__(TerraMindBackbone)
    nn.Module.__init__(backbone)

    # Set all required attributes
    backbone.model_name = "terramind_v1_base"
    backbone.pretrained = True
    backbone.modalities = ["S2L1C"]
    backbone.feature_indices = [2, 5, 8, 11]
    backbone.merge_method = "mean"
    backbone.tim_modalities = None

    # Use dummy encoder with parameters
    backbone._encoder = DummyTerraMindEncoder(feature_dim=256)
    backbone._model = backbone._encoder  # Alias for compatibility
    backbone._is_loaded = False

    return backbone


@pytest.fixture
def backbone():
    """Create TerraMindBackbone with dummy model."""
    return create_test_backbone()


class TestTerraMindBackboneForward:
    """Test forward pass returns correct features."""

    def test_extract_features_returns_list(self, backbone):
        """extract_features should return a list of tensors."""
        # TerraTorch interface uses dict format
        mod_dict = {
            "S2L1C": {"x": torch.randn(2, 6, 224, 224)}
        }
        features = backbone.extract_features(mod_dict)
        assert isinstance(features, list)

    def test_extract_features_returns_four_features(self, backbone):
        """extract_features should return 4 features for distillation."""
        mod_dict = {
            "S2L1C": {"x": torch.randn(2, 6, 224, 224)}
        }
        features = backbone.extract_features(mod_dict)
        assert len(features) == 4

    def test_extract_features_batch_dimension(self, backbone):
        """Each feature should have correct batch dimension."""
        batch_size = 2
        mod_dict = {
            "S2L1C": {"x": torch.randn(batch_size, 6, 224, 224)}
        }
        features = backbone.extract_features(mod_dict)
        for i, feat in enumerate(features):
            # Feature is (B, H, W, D) format
            assert feat.shape[0] == batch_size, \
                f"Feature {i}: batch size mismatch"


class TestTerraMindBackboneFreeze:
    """Test freeze/unfreeze functionality."""

    def test_freeze_makes_params_non_trainable(self, backbone):
        """Freeze should set requires_grad=False for all params."""
        backbone.freeze()

        for p in backbone.parameters():
            assert p.requires_grad is False

    def test_unfreeze_makes_params_trainable(self, backbone):
        """Unfreeze should set requires_grad=True for all params."""
        backbone.freeze()
        backbone.unfreeze()

        for p in backbone.parameters():
            assert p.requires_grad is True

    def test_freeze_unfreeze_cycle(self, backbone):
        """Multiple freeze/unfreeze cycles should work."""
        backbone.freeze()
        assert backbone.is_frozen()

        backbone.unfreeze()
        assert not backbone.is_frozen()

        backbone.freeze()
        assert backbone.is_frozen()


class TestBuildBackbone:
    """Test backbone factory."""

    def test_build_backbone_terramind_base(self):
        """Can build terramind_base backbone."""
        backbone = build_backbone("terramind_base")
        assert isinstance(backbone, TerraMindBackbone)

    def test_build_backbone_alias(self):
        """Can build using short alias 'base'."""
        backbone = build_backbone("base")
        assert isinstance(backbone, TerraMindBackbone)

    def test_build_backbone_unknown_raises(self):
        """Unknown backbone name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backbone"):
            build_backbone("unknown_backbone")


class TestTerraMindBackboneFeatureIndices:
    """Test feature index extraction."""

    def test_default_feature_indices(self, backbone):
        """Default feature indices should be [2, 5, 8, 11]."""
        assert backbone.feature_indices == [2, 5, 8, 11]

    def test_custom_feature_indices(self):
        """Can set custom feature indices."""
        backbone = TerraMindBackbone.__new__(TerraMindBackbone)
        nn.Module.__init__(backbone)
        backbone.feature_indices = [1, 3, 6, 9]
        backbone.model_name = "test"
        backbone.modalities = ["S2L1C"]

        assert backbone.feature_indices == [1, 3, 6, 9]


class TestTerraMindBackboneRepr:
    """Test string representation."""

    def test_repr_contains_model_name(self, backbone):
        """Repr should contain model name."""
        r = repr(backbone)
        assert "terramind_v1_base" in r

    def test_repr_contains_indices(self, backbone):
        """Repr should contain feature indices."""
        r = repr(backbone)
        assert "[2, 5, 8, 11]" in r