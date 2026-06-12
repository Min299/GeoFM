"""tests/test_phase4a_integration.py

Phase 4A Integration Test - Verifies all contracts work together.

This test exercises the complete integration pipeline:
    FeatureExtractorAdapter → ShapeValidator → ModelValidator → LoRAConfig → LoRATargets

If this passes, Phase 4A is complete and we can move to Phase 4B.
"""
import pytest
import torch
import torch.nn as nn


class DummyBackbone(nn.Module):
    """Dummy backbone for integration testing."""

    def extract_features(self, batch):
        return [
            torch.randn(2, 768, 64, 64),  # F2
            torch.randn(2, 768, 64, 64),  # F3 (extra)
            torch.randn(2, 768, 64, 64),  # F4 (extra)
            torch.randn(2, 768, 32, 32),  # F5
            torch.randn(2, 768, 32, 32),  # F6 (extra)
            torch.randn(2, 768, 32, 32),  # F7 (extra)
            torch.randn(2, 768, 16, 16),  # F8
            torch.randn(2, 768, 16, 16),  # F9 (extra)
            torch.randn(2, 768, 16, 16),  # F10 (extra)
            torch.randn(2, 768, 8, 8),     # F11
            torch.randn(2, 768, 8, 8),     # F12 (extra)
            torch.randn(2, 768, 8, 8),     # F13 (extra)
        ]

    def parameters(self, recurse=True):
        return []  # No trainable params


class DummyAdapterBank(nn.Module):
    """Dummy adapter bank."""

    def __init__(self):
        super().__init__()
        self.adapters = nn.ModuleDict()

    def __contains__(self, key):
        return key in self.adapters

    def forward(self, task, features):
        return features


class DummyDecoderBank(nn.Module):
    """Dummy decoder bank."""

    def __init__(self):
        super().__init__()
        self.decoders = nn.ModuleDict({
            "flood": nn.Identity(),
            "burn": nn.Identity(),
            "lulc": nn.Identity(),
        })

    def __contains__(self, key):
        return key in self.decoders

    def __getitem__(self, key):
        return self.decoders[key]

    def keys(self):
        return self.decoders.keys()

    def forward(self, task, features):
        return torch.randn(2, 2, 64, 64)


class DummySharedGeoFM(nn.Module):
    """Dummy SharedGeoFM model."""

    def __init__(self):
        super().__init__()
        self.backbone = DummyBackbone()
        self.adapter_bank = DummyAdapterBank()
        self.decoder_bank = DummyDecoderBank()

    def forward(self, batch, task_name):
        features = self.backbone.extract_features(batch)
        features = self.adapter_bank(task_name, features)
        return self.decoder_bank(task_name, features)


class TestPhase4AIntegration:
    """Integration tests for Phase 4A."""

    def test_fextractor_to_shape_validator(self):
        """FeatureExtractorAdapter → ShapeValidator pipeline."""
        from geofm.integration import FeatureExtractorAdapter, ShapeValidator

        backbone = DummyBackbone()
        extractor = FeatureExtractorAdapter(backbone)

        # Extract features
        features = extractor.extract({})

        # Validate shapes
        ShapeValidator.validate_all(features)

        assert len(features) == 4
        assert all(f.shape[1] == 768 for f in features)

    def test_shape_validator_to_model_validator(self):
        """ShapeValidator → ModelValidator pipeline."""
        from geofm.integration import ShapeValidator, ModelValidator

        model = DummySharedGeoFM()

        # Get features with correct indices (extract 4 features at indices [2,5,8,11])
        features = [
            model.backbone.extract_features({})[2],
            model.backbone.extract_features({})[5],
            model.backbone.extract_features({})[8],
            model.backbone.extract_features({})[11],
        ]

        # Validate shapes
        ShapeValidator.validate_all(features)

        # Validate model structure
        ModelValidator.validate_model(model)

        # Validate task
        ModelValidator.validate_task(model, "flood")

    def test_lora_config_integration(self):
        """LoRAConfig integration."""
        from geofm.models.peft.lora_config import LoRAConfig, get_lora_config

        # Create custom config
        config = LoRAConfig(rank=16, alpha=32)

        # Verify scale
        assert config.scale == 2.0

        # Get preset
        base_config = get_lora_config("base")
        assert base_config.rank == 16

    def test_lora_targets_integration(self):
        """LoRATargets integration."""
        from geofm.models.peft.lora_targets import (
            get_lora_targets,
            get_target_names,
            DEFAULT_LORA_TARGETS,
        )

        # Get TerraMind targets
        targets = get_lora_targets("terramind")
        assert "attn" in targets.qkv

        # Get target names
        names = get_target_names(targets, include_mlp=False)
        assert len(names) == 2

        # Verify default
        assert DEFAULT_LORA_TARGETS is not None

    def test_end_to_end_pipeline(self):
        """Complete pipeline: Extractor → Validator → Config → Targets."""
        from geofm.integration import (
            FeatureExtractorAdapter,
            ShapeValidator,
            ModelValidator,
        )
        from geofm.models.peft.lora_config import LoRAConfig
        from geofm.models.peft.lora_targets import get_lora_targets, get_target_names

        # 1. Create model
        model = DummySharedGeoFM()

        # 2. Create extractor
        extractor = FeatureExtractorAdapter(model.backbone)

        # 3. Extract features (with correct indices)
        all_features = model.backbone.extract_features({})
        features = [all_features[i] for i in extractor.get_indices()]

        # 4. Validate shapes
        ShapeValidator.validate_all(features)

        # 5. Validate model
        ModelValidator.validate_all(model, task="flood")

        # 6. Get LoRA config
        config = LoRAConfig(rank=8, alpha=16)
        assert config.scale == 2.0

        # 7. Get LoRA targets
        targets = get_lora_targets("terramind")
        target_names = get_target_names(targets, include_mlp=True)
        assert len(target_names) >= 2

    def test_model_validation_all_tasks(self):
        """Validate all registered tasks."""
        from geofm.integration import ModelValidator

        model = DummySharedGeoFM()

        for task in ["flood", "burn", "lulc"]:
            ModelValidator.validate_task(model, task)

    def test_shape_validation_different_batch_sizes(self):
        """ShapeValidator with different configurations."""
        from geofm.integration import ShapeValidator

        # Different batch size
        features = [
            torch.randn(4, 768, 64, 64),
            torch.randn(4, 768, 32, 32),
            torch.randn(4, 768, 16, 16),
            torch.randn(4, 768, 8, 8),
        ]

        ShapeValidator.validate_all(features)

        info = ShapeValidator.get_feature_info(features)
        assert info["batch_size"] == 4
        assert info["channels"] == 768

    def test_lora_config_presets(self):
        """Test all LoRA presets."""
        from geofm.models.peft.lora_config import LORA_CONFIGS

        for name, config in LORA_CONFIGS.items():
            assert config.rank > 0
            assert config.alpha > 0

    def test_integration_constants(self):
        """Test integration module constants."""
        from geofm.integration import (
            BackboneAdapter,
            FeatureExtractorAdapter,
            ShapeValidator,
            ModelValidator,
        )

        # All should be importable
        assert BackboneAdapter is not None
        assert FeatureExtractorAdapter is not None
        assert ShapeValidator is not None
        assert ModelValidator is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])