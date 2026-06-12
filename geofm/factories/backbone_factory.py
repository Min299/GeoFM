"""geofm.factories.backbone_factory

Factory for creating backbone encoders.
Supports TerraMind (base, small, tiny), Prithvi, StudentViT, and other backbones.
"""
from __future__ import annotations

from typing import Optional, Dict, Any, Type


class BackboneFactory:
    """Factory for creating backbone encoders.

    This factory supports multiple backbone types:
    - TerraMind: Primary backbone for GeoFM (base, small, tiny variants)
    - Prithvi: Temporal/spatial backbone for flood
    - StudentViT: Distilled student model
    - Generic: Any model conforming to BackboneAdapter interface

    Usage:
        # Create TerraMind backbone
        backbone = BackboneFactory.build("terramind_base")

        # Create TerraMind small variant
        backbone = BackboneFactory.build("terramind_small")

        # Create generic ViT
        backbone = BackboneFactory.build("vit", img_size=224, patch_size=16)
    """

    _registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str, backbone_class: Type) -> None:
        """Register a new backbone type.

        Args:
            name: Name to register (e.g., "terramind", "prithvi")
            backbone_class: Class for the backbone
        """
        cls._registry[name.lower()] = backbone_class

    @classmethod
    def build(
        cls,
        backbone_name: str,
        **kwargs,
    ) -> Any:
        """Build a backbone by name.

        Args:
            backbone_name: Name of the backbone (terramind_base, terramind_small, prithvi, vit, etc.)
            **kwargs: Arguments to pass to the backbone constructor

        Returns:
            Backbone instance

        Raises:
            ValueError: If backbone name is unknown
        """
        name = backbone_name.lower()

        # Check registry first
        if name in cls._registry:
            return cls._registry[name](**kwargs)

        # TerraMind variants
        if name.startswith("terramind"):
            return cls._build_terramind(name, **kwargs)

        if name == "prithvi":
            return cls._build_prithvi(**kwargs)

        if name in ("vit", "vit_base", "vit_small", "vit_tiny"):
            return cls._build_vit(name, **kwargs)

        if name == "resnet":
            return cls._build_resnet(**kwargs)

        if name.startswith("student"):
            return cls._build_student_vit(name, **kwargs)

        raise ValueError(
            f"Unknown backbone: {backbone_name}. "
            f"Available: {list(cls._registry.keys()) + ['terramind_base', 'terramind_small', 'terramind_tiny', 'prithvi', 'vit', 'resnet', 'student_vit']}"
        )

    @classmethod
    def _build_terramind(cls, name: str, **kwargs) -> Any:
        """Build TerraMind backbone.

        Supports variants:
        - terramind_base: Full model
        - terramind_small: Smaller variant
        - terramind_tiny: Tiny variant

        Args:
            name: TerraMind variant name
            **kwargs: TerraMind-specific arguments

        Returns:
            TerraMind backbone
        """
        try:
            from geofm.models.backbones.terramind_backbone import TerraMindBackbone

            # Map variant name to model name
            variant_map = {
                "terramind_base": "terramind_v1_base",
                "terramind_small": "terramind_v1_small",
                "terramind_tiny": "terramind_v1_tiny",
            }

            model_name = variant_map.get(name, name)

            return TerraMindBackbone(
                model_name=model_name,
                **kwargs,
            )
        except ImportError:
            raise ImportError(
                "TerraMind backbone not available. "
                "Ensure geofm.models.backbones.terramind_backbone is installed."
            )

    @classmethod
    def _build_prithvi(cls, **kwargs) -> Any:
        """Build Prithvi backbone.

        Args:
            **kwargs: Prithvi-specific arguments

        Returns:
            Prithvi backbone
        """
        # Prithvi integration would go here
        raise NotImplementedError(
            "Prithvi backbone not yet integrated. "
            "Use 'terramind_base', 'terramind_small', or 'vit' for now."
        )

    @classmethod
    def _build_vit(cls, name: str, **kwargs) -> Any:
        """Build ViT backbone.

        Args:
            name: ViT variant name
            **kwargs: ViT-specific arguments

        Returns:
            ViT backbone
        """
        try:
            from transformers import ViTModel

            # Map variant names to configs
            variant_map = {
                "vit": "google/vit-base-patch16-224",
                "vit_base": "google/vit-base-patch16-224",
                "vit_small": "google/vit-small-patch16-224",
                "vit_tiny": "google/vit-tiny-patch16-224",
            }

            model_name = variant_map.get(name, "google/vit-base-patch16-224")

            return ViTModel.from_pretrained(model_name)
        except ImportError:
            raise ImportError(
                "Transformers library not available. "
                "Install with: pip install transformers"
            )

    @classmethod
    def _build_resnet(cls, **kwargs) -> Any:
        """Build ResNet backbone.

        Args:
            **kwargs: ResNet-specific arguments

        Returns:
            ResNet backbone
        """
        import torchvision.models as models

        variant = kwargs.pop("variant", "resnet50")
        pretrained = kwargs.pop("pretrained", True)

        variant_map = {
            "resnet18": (models.resnet18, pretrained),
            "resnet34": (models.resnet34, pretrained),
            "resnet50": (models.resnet50, pretrained),
            "resnet101": (models.resnet101, pretrained),
        }

        if variant not in variant_map:
            raise ValueError(f"Unknown ResNet variant: {variant}")

        model_fn, pretrain = variant_map[variant]
        return model_fn(pretrained=pretrain)

    @classmethod
    def _build_student_vit(cls, name: str, **kwargs) -> Any:
        """Build Student ViT for distillation.

        Args:
            name: Student variant name
            **kwargs: Student-specific arguments

        Returns:
            Student ViT backbone
        """
        try:
            from geofm.models.backbones.student_vit import StudentViT

            return StudentViT(**kwargs)
        except ImportError:
            raise NotImplementedError(
                "StudentViT not available. "
                "Use 'terramind_base' or 'vit' for now."
            )

    @classmethod
    def list_available(cls) -> list:
        """List available backbone types.

        Returns:
            List of available backbone names
        """
        available = list(cls._registry.keys()) + [
            "terramind_base",
            "terramind_small",
            "terramind_tiny",
            "prithvi",
            "vit",
            "resnet",
            "student_vit",
        ]
        return sorted(set(available))