"""tests/test_model_builder.py

Tests for model builder.
"""
import pytest
import torch.nn as nn


class TestModelBuilder:
    """Test ModelBuilder class."""

    def test_builder_exists(self):
        """ModelBuilder should be importable."""
        from geofm.builders import ModelBuilder

        assert ModelBuilder is not None

    def test_builder_init(self):
        """ModelBuilder should initialize."""
        from geofm.builders import ModelBuilder

        builder = ModelBuilder()

        assert builder is not None

    def test_build_shared_model(self):
        """Should build shared model."""
        from geofm.builders import ModelBuilder

        builder = ModelBuilder()

        backbone = nn.Linear(10, 10)
        adapter_bank = nn.Linear(10, 10)
        decoder_bank = nn.Linear(10, 10)

        model = builder.build_shared_model(
            backbone=backbone,
            adapter_bank=adapter_bank,
            decoder_bank=decoder_bank,
        )

        assert model is not None
        assert model.backbone is backbone
        assert model.adapter_bank is adapter_bank
        assert model.decoder_bank is decoder_bank

    def test_get_model_info(self):
        """Should return model info."""
        from geofm.builders import ModelBuilder

        builder = ModelBuilder()

        model = nn.Linear(10, 5)

        info = builder.get_model_info(model)

        assert "total_parameters" in info
        assert "trainable_parameters" in info
        assert "frozen_parameters" in info
        assert "trainable_percentage" in info

        assert info["total_parameters"] == 55  # 10*5 + 5

    def test_build_from_config_raises(self):
        """Config building should raise NotImplementedError."""
        from geofm.builders import ModelBuilder

        builder = ModelBuilder()

        with pytest.raises(NotImplementedError):
            builder.build_from_config({})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])