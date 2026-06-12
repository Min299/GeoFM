"""geofm.models.backbones

Backbone models: TerraMind wrapper and factory.
"""
from geofm.models.backbones.backbone_factory import (
    build_backbone,
    list_available_backbones,
)
from geofm.models.backbones.terramind_factory import (
    TerraMindConfig,
    create_terramind_config,
    get_terramind_info,
    list_available_variants,
    TERRAMIND_VARIANTS,
)
from geofm.models.backbones.terramind_backbone import TerraMindBackbone
from geofm.models.backbones.terramind_wrapper import TerraMindWrapper

__all__ = [
    # Core backbone
    "TerraMindBackbone",
    "TerraMindWrapper",
    # Factory
    "build_backbone",
    "list_available_backbones",
    # Config
    "TerraMindConfig",
    "create_terramind_config",
    "get_terramind_info",
    "list_available_variants",
    "TERRAMIND_VARIANTS",
]
