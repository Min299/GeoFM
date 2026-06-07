"""geofm.models.backbones.terramind_backbone

TerraMind backbone wrapper for GeoFM.
Wraps TerraTorch's TerraMind with proper interface.

Key findings from TerraTorch source:
- encoder returns List[Tensor] - one per transformer layer
- Shape: (B, N, D) where N=tokens, D=dim
- SelectIndices neck extracts specific layers [2, 5, 8, 11]
- Multi-modality: mod_dict[str, dict[str, Tensor]]

Based on:
  terratorch.models.backbones.terramind.model.terramind.TerraMind
  terratorch.models.necks.SelectIndices
"""
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

import torch
import torch.nn as nn


class TerraMindBackbone(nn.Module):
    """Wrapper around TerraTorch's TerraMind backbone.

    Handles modality-based inputs and provides feature extraction
    for decoder heads and distillation.

    Usage:
        # Create from config
        backbone = TerraMindBackbone.from_config(
            model_name="terramind_v1_base",
            modalities=["S2L1C", "S1GRD"]
        )

        # Forward pass with modalities dict
        mod_dict = {
            "S2L1C": {"x": s2_tensor, "input_mask": mask},
            "S1GRD": {"x": sar_tensor, "input_mask": mask}
        }
        features = backbone(mod_dict)
        # features is List[Tensor] - one per layer, shape (B, N, D)

        # Extract specific feature levels for distillation
        f2, f5, f8, f11 = backbone.extract_features(mod_dict, indices=[2, 5, 8, 11])
    """

    DEFAULT_FEATURE_INDICES = [2, 5, 8, 11]

    def __init__(
        self,
        model_name: str = "terramind_v1_base",
        pretrained: bool = True,
        modalities: List[str] = None,
        feature_indices: List[int] = None,
        merge_method: str = "mean",
        tim_modalities: Optional[List[str]] = None,
    ):
        super().__init__()

        self.model_name = model_name
        self.pretrained = pretrained
        self.modalities = modalities or ["S2L1C"]
        self.feature_indices = feature_indices or self.DEFAULT_FEATURE_INDICES
        self.merge_method = merge_method
        self.tim_modalities = tim_modalities

        # Load the actual TerraMind model
        self._load_terramind()

    def _load_terramind(self):
        """Load TerraMind from terratorch registry."""
        from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY

        # Map model_name to registry name
        registry_name = self.model_name
        if not registry_name.startswith("terramind_"):
            registry_name = f"terramind_v1_{model_name}"

        # TerraMind expects band configuration to match pretrained weights
        # S2L1C has 13 bands: COASTAL_AEROSOL, BLUE, GREEN, RED, RED_EDGE_1, RED_EDGE_2,
        # RED_EDGE_3, NIR_BROAD, NIR_NARROW, WATER_VAPOR, CIRRUS, SWIR_1, SWIR_2
        S2L1C_BANDS = [
            "COASTAL_AEROSOL", "BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2",
            "RED_EDGE_3", "NIR_BROAD", "NIR_NARROW", "WATER_VAPOR", "CIRRUS", "SWIR_1", "SWIR_2"
        ]

        # S2L2A has 12 bands (no CIRRUS)
        S2L2A_BANDS = [
            "COASTAL_AEROSOL", "BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2",
            "RED_EDGE_3", "NIR_BROAD", "NIR_NARROW", "WATER_VAPOR", "SWIR_1", "SWIR_2"
        ]

        # Select bands based on modality
        if "S2L1C" in self.modalities:
            bands = {"image": S2L1C_BANDS}
        elif "S2L2A" in self.modalities:
            bands = {"image": S2L2A_BANDS}
        else:
            bands = None

        # Build model from registry
        # Note: pretrained_bands is already included in the registry functions
        self._model = TERRATORCH_BACKBONE_REGISTRY.build(
            registry_name,
            pretrained=self.pretrained,
            bands=bands,
        )

        # Store feature dim (depends on model variant)
        # TerraMindViT has out_channels list - use first layer's channel count
        self._feature_dim = self._model.out_channels[0] if hasattr(self._model, 'out_channels') else 768
        self._num_layers = len(self._model.out_channels) if hasattr(self._model, 'out_channels') else 12

    @classmethod
    def from_config(cls, config) -> "TerraMindBackbone":
        """Create backbone from TerraMindConfig."""
        return cls(
            model_name=config.model_name,
            pretrained=config.pretrained,
            modalities=config.modalities,
            feature_indices=config.feature_indices,
            merge_method=config.merge_method,
            tim_modalities=config.tim_modalities,
        )

    def freeze(self):
        """Freeze all backbone parameters."""
        if self._model is not None:
            for p in self._model.parameters():
                p.requires_grad = False

    def unfreeze(self):
        """Unfreeze all backbone parameters."""
        if self._model is not None:
            for p in self._model.parameters():
                p.requires_grad = True

    def is_frozen(self) -> bool:
        """Check if backbone is frozen."""
        if self._model is None:
            return False
        for p in self._model.parameters():
            if p.requires_grad:
                return False
        return True

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through the backbone.

        Args:
            x: Input tensor (B, C, H, W) - typically (B, 13, 224, 224) for S2L1C
               OR dict format: {"image": tensor} for TerraTorch compatibility

        Returns:
            List of feature tensors from each transformer layer.
            Shape per tensor: (B, N, D) where N=sequence_length, D=embed_dim
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call _load_terramind() first.")

        # Handle both dict and tensor inputs
        if isinstance(x, dict):
            # Already in TerraTorch dict format
            mod_dict = x
        else:
            # Convert tensor to dict format
            mod_dict = {"image": x}

        # TerraMind expects dict format: {modality: tensor}
        features = self._model(mod_dict)

        return features

    def extract_features(
        self,
        x: torch.Tensor,
        indices: Optional[List[int]] = None
    ) -> List[torch.Tensor]:
        """Extract features at specific indices for distillation.

        Args:
            x: Input tensor (B, C, H, W)
            indices: Feature indices to extract [2, 5, 8, 11] default

        Returns:
            List of feature tensors at specified indices
            Shape per tensor: (B, N, D) where N=sequence_length, D=embed_dim
        """
        indices = indices or self.feature_indices
        all_features = self.forward(x)

        return [all_features[i] for i in indices]

    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about available feature indices."""
        return {
            "model_name": self.model_name,
            "modalities": self.modalities,
            "feature_indices": self.feature_indices,
            "merge_method": self.merge_method,
            "feature_dim": self._feature_dim,
            "num_layers": self._num_layers,
            "output_type": "List[Tensor]",
            "output_shape": "(B, N, D) - Batch, Sequence, Embed",
        }

    def __repr__(self):
        return (
            f"TerraMindBackbone(\n"
            f"  model_name={self.model_name},\n"
            f"  modalities={self.modalities},\n"
            f"  feature_indices={self.feature_indices},\n"
            f"  merge_method={self.merge_method},\n"
            f"  feature_dim={self._feature_dim},\n"
            f"  num_layers={self._num_layers}\n"
            f")"
        )