"""geofm.models

Model components: backbones, features, decoders, heads, LoRA.
"""
from geofm.models.backbones.terramind_factory import (
    TerraMindConfig,
    create_terramind_config,
    get_terramind_info,
    list_available_variants,
    TERRAMIND_VARIANTS,
)
from geofm.models.backbones.terramind_backbone import TerraMindBackbone
from geofm.models.features.feature_extractor import (
    FeatureExtractor,
    FeatureLevels,
    DistillationLoss,
    create_feature_extractor,
)

__all__ = [
    # Factory
    "TerraMindConfig",
    "create_terramind_config",
    "get_terramind_info",
    "list_available_variants",
    "TERRAMIND_VARIANTS",
    # Backbone
    "TerraMindBackbone",
    # Features
    "FeatureExtractor",
    "FeatureLevels",
    "DistillationLoss",
    "create_feature_extractor",
]
