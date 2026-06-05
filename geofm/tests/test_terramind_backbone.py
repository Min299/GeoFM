"""Contract tests for TerraMind backbone and feature extractor.

Validates the interface based on actual TerraTorch API:
- Input: mod_dict[str, dict[str, Tensor]]
- Output: List[Tensor] with shape (B, N, D)
- Feature indices: [2, 5, 8, 11] for base/tiny/small, [5, 11, 17, 23] for large
"""
import pytest
import torch

from geofm.models import (
    TerraMindConfig,
    TerraMindBackbone,
    FeatureExtractor,
    FeatureLevels,
    DistillationLoss,
    create_terramind_config,
    get_terramind_info,
    list_available_variants,
)


class TestTerraMindConfig:
    def test_default_config(self):
        config = TerraMindConfig()
        assert config.model_name == "terramind_v1_base"
        assert config.modalities == ["S2L1C"]
        assert config.feature_indices == [2, 5, 8, 11]

    def test_create_terramind_config(self):
        config = create_terramind_config(
            model_name="terramind_v1_tiny",
            modalities=["S2L1C", "S1GRD"]
        )
        assert config.model_name == "terramind_v1_tiny"
        assert config.modalities == ["S2L1C", "S1GRD"]
        assert config.feature_indices == [2, 5, 8, 11]  # From variant info

    def test_large_model_indices(self):
        config = create_terramind_config(model_name="terramind_v1_large")
        assert config.feature_indices == [5, 11, 17, 23]

    def test_get_terramind_info(self):
        info = get_terramind_info("terramind_v1_base")
        assert "params" in info
        assert "indices" in info
        assert info["params"] == "~87M"

    def test_list_available_variants(self):
        variants = list_available_variants()
        assert "terramind_v1_tiny" in variants
        assert "terramind_v1_base" in variants
        assert "terramind_v1_large" in variants


class TestTerraMindBackbone:
    def test_backbone_creation(self):
        backbone = TerraMindBackbone(
            model_name="terramind_v1_base",
            modalities=["S2L1C"]
        )
        assert backbone.model_name == "terramind_v1_base"
        assert backbone.modalities == ["S2L1C"]
        assert backbone.feature_indices == [2, 5, 8, 11]

    def test_backbone_from_config(self):
        config = TerraMindConfig(
            model_name="terramind_v1_tiny",
            modalities=["S2L1C", "S1GRD"],
            feature_indices=[2, 5, 8, 11]
        )
        backbone = TerraMindBackbone.from_config(config)
        assert backbone.model_name == "terramind_v1_tiny"
        assert backbone.modalities == ["S2L1C", "S1GRD"]

    def test_forward_placeholder(self):
        backbone = TerraMindBackbone()

        # TerraTorch format input
        mod_dict = {
            "S2L1C": {"x": torch.randn(1, 13, 224, 224)}
        }

        features = backbone(mod_dict)
        assert isinstance(features, list)
        assert len(features) > 0
        assert features[0].ndim == 3  # (B, N, D)

    def test_extract_features(self):
        backbone = TerraMindBackbone(feature_indices=[2, 5, 8, 11])

        mod_dict = {
            "S2L1C": {"x": torch.randn(1, 13, 224, 224)}
        }

        features = backbone.extract_features(mod_dict)
        assert len(features) == 4  # [2, 5, 8, 11]
        for f in features:
            assert f.ndim == 3

    def test_freeze_unfreeze(self):
        backbone = TerraMindBackbone()
        backbone.freeze()
        # Check that parameters are frozen (requires_grad = False)
        frozen_count = sum(1 for p in backbone.parameters() if not p.requires_grad)
        total_count = sum(1 for _ in backbone.parameters())
        assert frozen_count == total_count, "Not all params are frozen"
        backbone.unfreeze()
        # Check that parameters are unfrozen
        unfrozen_count = sum(1 for p in backbone.parameters() if p.requires_grad)
        assert unfrozen_count == total_count, "Not all params are unfrozen"

    def test_get_feature_info(self):
        backbone = TerraMindBackbone()
        info = backbone.get_feature_info()
        assert "model_name" in info
        assert "feature_indices" in info
        assert "output_type" in info
        assert info["output_type"] == "List[Tensor]"


class TestFeatureLevels:
    def test_from_list(self):
        features = [torch.randn(2, 256, 768) for _ in range(4)]
        indices = [2, 5, 8, 11]

        levels = FeatureLevels.from_list(features, indices)
        assert levels.f2 is not None
        assert levels.f5 is not None
        assert levels.f8 is not None
        assert levels.f11 is not None

    def test_to_list(self):
        levels = FeatureLevels(f2=torch.randn(1, 256, 768))
        result = levels.to_list()
        assert len(result) == 1

    def test_iter(self):
        levels = FeatureLevels(
            f2=torch.randn(1, 256, 768),
            f5=torch.randn(1, 256, 768)
        )
        count = sum(1 for _ in levels)
        assert count == 2


class TestFeatureExtractor:
    def test_creation(self):
        backbone = TerraMindBackbone()
        extractor = FeatureExtractor(backbone)
        assert extractor.indices == [2, 5, 8, 11]

    def test_extraction(self):
        backbone = TerraMindBackbone(feature_indices=[2, 5, 8, 11])
        extractor = FeatureExtractor(backbone)

        mod_dict = {"S2L1C": {"x": torch.randn(1, 13, 224, 224)}}
        levels = extractor(mod_dict)

        assert isinstance(levels, FeatureLevels)
        assert levels.f2 is not None or levels.f5 is not None

    def test_create_feature_extractor(self):
        from geofm.models.features.feature_extractor import create_feature_extractor

        backbone = TerraMindBackbone(model_name="terramind_v1_base")
        extractor = create_feature_extractor(backbone, "terramind_v1_base")
        assert extractor.indices == [2, 5, 8, 11]

        # Large model should have different indices
        backbone_large = TerraMindBackbone(model_name="terramind_v1_large")
        extractor_large = create_feature_extractor(backbone_large, "terramind_v1_large")
        assert extractor_large.indices == [5, 11, 17, 23]


class TestDistillationLoss:
    def test_creation(self):
        loss_fn = DistillationLoss()
        assert loss_fn.indices == [2, 5, 8, 11]

    def test_forward(self):
        loss_fn = DistillationLoss(indices=[2, 5])

        teacher = FeatureLevels(
            f2=torch.randn(1, 256, 768),
            f5=torch.randn(1, 256, 768)
        )
        student = FeatureLevels(
            f2=torch.randn(1, 256, 768),
            f5=torch.randn(1, 256, 768)
        )

        loss = loss_fn(teacher, student)
        assert loss.item() >= 0

    def test_shape_mismatch_handling(self):
        loss_fn = DistillationLoss(indices=[2])

        teacher = FeatureLevels(f2=torch.randn(1, 256, 768))
        student = FeatureLevels(f2=torch.randn(1, 128, 512))  # Different shape

        loss = loss_fn(teacher, student)
        assert loss.item() >= 0  # Should handle alignment