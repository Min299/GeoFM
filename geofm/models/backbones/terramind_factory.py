"""geofm.models.backbones.terramind_factory

Factory for creating TerraMind backbones via TerraTorch's EncoderDecoderFactory.
Based on real TerraTorch config patterns from IBM's flood/burn/crop experiments.
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class TerraMindConfig:
    """Configuration for TerraMind backbone.

    Based on TerraTorch's EncoderDecoderFactory configuration format.

    Example usage:
        config = TerraMindConfig(
            model_name="terramind_v1_base",
            pretrained=True,
            modalities=["S2L1C", "S1GRD"],
            feature_indices=[2, 5, 8, 11],  # For distillation
        )
        backbone = create_terramind(config)
    """
    model_name: str = "terramind_v1_base"
    pretrained: bool = True
    modalities: List[str] = None  # e.g., ["S2L1C"], ["S2L1C", "S1GRD"]
    bands: Optional[Dict[str, List[str]]] = None  # e.g., {"S2L2A": ["BLUE", "GREEN", "RED"]}
    feature_indices: List[int] = None  # [2, 5, 8, 11] for distillation
    tim_modalities: Optional[List[str]] = None  # For TIM models, e.g., ["LULC"]
    merge_method: str = "mean"  # How to merge modalities: "mean", "concat"

    def __post_init__(self):
        if self.modalities is None:
            self.modalities = ["S2L1C"]
        if self.feature_indices is None:
            self.feature_indices = [2, 5, 8, 11]


# Supported TerraMind model variants
TERRAMIND_VARIANTS = {
    "terramind_v1_tiny": {
        "params": "~6M",
        "indices": [2, 5, 8, 11],
        "description": "Tiny variant for fast experiments",
    },
    "terramind_v1_small": {
        "params": "~22M",
        "indices": [2, 5, 8, 11],
        "description": "Small variant",
    },
    "terramind_v1_base": {
        "params": "~87M",
        "indices": [2, 5, 8, 11],
        "description": "Base variant - recommended for research",
    },
    "terramind_v1_base_tim": {
        "params": "~87M + TIM",
        "indices": [2, 5, 8, 11],
        "description": "Base with Text-Image Matching",
    },
    "terramind_v1_large": {
        "params": "~300M",
        "indices": [5, 11, 17, 23],
        "description": "Large variant - expensive",
    },
}


def create_terramind_config(
    model_name: str = "terramind_v1_base",
    modalities: List[str] = None,
    **kwargs
) -> TerraMindConfig:
    """Create a TerraMindConfig with smart defaults.

    Args:
        model_name: Model variant (tiny, small, base, base_tim, large)
        modalities: List of modalities (S2L1C, S2L2A, S1GRD, RGB, DEM)
        **kwargs: Additional config overrides

    Returns:
        TerraMindConfig instance
    """
    # Auto-set feature indices based on model variant
    variant_info = TERRAMIND_VARIANTS.get(model_name, {})
    default_indices = variant_info.get("indices", [2, 5, 8, 11])

    return TerraMindConfig(
        model_name=model_name,
        pretrained=True,
        modalities=modalities or ["S2L1C"],
        feature_indices=default_indices,
        **kwargs
    )


def get_terramind_info(model_name: str) -> Dict[str, Any]:
    """Get information about a TerraMind variant.

    Args:
        model_name: Model variant name

    Returns:
        Dict with params, indices, description
    """
    return TERRAMIND_VARIANTS.get(model_name, {})


def list_available_variants() -> List[str]:
    """List all available TerraMind variants."""
    return list(TERRAMIND_VARIANTS.keys())