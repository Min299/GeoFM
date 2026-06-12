"""geofm.models.backbones.backbone_factory

Factory for building GeoFM backbones.

Centralizes backbone instantiation to ensure consistent setup
across all experiments.
"""
from __future__ import annotations

from geofm.models.backbones.terramind_backbone import (
    TerraMindBackbone,
)


def build_backbone(
    backbone_name: str,
    pretrained: bool = True,
    **kwargs,
):
    """Build a backbone by name.

    Uses lazy instantiation - only creates the requested backbone.

    Args:
        backbone_name: Name of the backbone variant.
            Options: terramind_tiny, terramind_small, terramind_base
        pretrained: Whether to load pretrained weights.
        **kwargs: Additional arguments passed to the backbone.

    Returns:
        Initialized backbone module.

    Raises:
        ValueError: If backbone_name is not recognized.
    """
    registry = {

        "terramind_tiny":
        lambda: TerraMindBackbone(
            model_name="terramind_v1_tiny",
            pretrained=pretrained,
            **kwargs,
        ),

        "terramind_small":
        lambda: TerraMindBackbone(
            model_name="terramind_v1_small",
            pretrained=pretrained,
            **kwargs,
        ),

        "terramind_base":
        lambda: TerraMindBackbone(
            model_name="terramind_v1_base",
            pretrained=pretrained,
            **kwargs,
        ),

        # Aliases for convenience
        "tiny":
        lambda: TerraMindBackbone(
            model_name="terramind_v1_tiny",
            pretrained=pretrained,
            **kwargs,
        ),

        "small":
        lambda: TerraMindBackbone(
            model_name="terramind_v1_small",
            pretrained=pretrained,
            **kwargs,
        ),

        "base":
        lambda: TerraMindBackbone(
            model_name="terramind_v1_base",
            pretrained=pretrained,
            **kwargs,
        ),
    }

    if backbone_name not in registry:
        available = list(registry.keys())
        raise ValueError(
            f"Unknown backbone: {backbone_name!r}\n"
            f"Available backbones: {available}"
        )

    # Lazy instantiation - only call the lambda when needed
    return registry[backbone_name]()


def list_available_backbones():
    """List all available backbone variants."""
    return [
        "terramind_tiny",
        "terramind_small",
        "terramind_base",
        "tiny",
        "small",
        "base",
    ]