"""tests/test_shared_model_builder.py

Tests for SharedModelBuilder.
"""
import pytest
import torch
import torch.nn as nn


class TestSharedModelBuilder:
    """Test SharedModelBuilder class."""

    def test_builder_init(self):
        """Builder should initialize."""
        from geofm.builders.shared_model_builder import SharedModelBuilder

        builder = SharedModelBuilder()

        assert builder is not None

    def test_builder_with_components(self):
        """Builder should accept components."""
        from geofm.builders.shared_model_builder import SharedModelBuilder

        builder = SharedModelBuilder(
            backbone=nn.Identity(),
            adapter_bank=nn.Identity(),
            decoder_bank=nn.Identity(),
        )

        assert builder.backbone is not None
        assert builder.adapter_bank is not None
        assert builder.decoder_bank is not None

    def test_builder_setters(self):
        """Builder should have setter methods."""
        from geofm.builders.shared_model_builder import SharedModelBuilder

        builder = SharedModelBuilder()

        builder.set_backbone(nn.Linear(10, 10))
        builder.set_adapter_bank(nn.Linear(10, 10))
        builder.set_decoder_bank(nn.Linear(10, 10))

        assert builder.backbone is not None
        assert builder.adapter_bank is not None
        assert builder.decoder_bank is not None

    def test_builder_build(self):
        """build should create SharedGeoFM."""
        from geofm.builders.shared_model_builder import SharedModelBuilder
        from geofm.models.multitask.shared_geofm import SharedGeoFM

        builder = SharedModelBuilder(
            backbone=nn.Identity(),
            adapter_bank=nn.Identity(),
            decoder_bank=nn.Identity(),
        )

        model = builder.build()

        assert isinstance(model, SharedGeoFM)

    def test_builder_missing_backbone(self):
        """Build should fail without backbone."""
        from geofm.builders.shared_model_builder import SharedModelBuilder

        builder = SharedModelBuilder(
            adapter_bank=nn.Identity(),
            decoder_bank=nn.Identity(),
        )

        with pytest.raises(ValueError, match="Backbone must be set"):
            builder.build()

    def test_builder_get_info(self):
        """get_model_info should return info."""
        from geofm.builders.shared_model_builder import SharedModelBuilder

        builder = SharedModelBuilder(
            backbone=nn.Linear(10, 10),
            adapter_bank=nn.Identity(),
            decoder_bank=nn.Identity(),
        )

        info = builder.get_model_info()

        assert "has_backbone" in info
        assert "has_adapter_bank" in info
        assert "has_decoder_bank" in info
        assert info["has_backbone"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])