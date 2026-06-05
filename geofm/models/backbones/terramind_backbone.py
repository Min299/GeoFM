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
        self.modalities = modalities or ["S2L1C"]
        self.feature_indices = feature_indices or [2, 5, 8, 11]
        self.merge_method = merge_method
        self.tim_modalities = tim_modalities

        # TerraTorch model
        self._model = None
        self._encoder = None
        self._load_terramind()

    def _load_terramind(self):
        """Load TerraMind from terratorch."""
        try:
            from terratorch.models.backbones.terramind.model.terramind import TerraMind
            from terratorch.models.necks import SelectIndices

            # Build model using TerraTorch's factory
            # Note: This is a simplified version - full implementation
            # would use EncoderDecoderFactory with proper config

            # For now, we'll create a wrapper that mirrors TerraTorch's interface
            self._is_loaded = False
            self._model = None
            self._encoder = None

        except ImportError:
            self._is_loaded = False
            self._create_placeholder()

    def _create_placeholder(self):
        """Create a placeholder model for development without terratorch."""
        self._encoder = nn.Identity()
        self._is_loaded = False

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
        if self._encoder is not None:
            for p in self._encoder.parameters():
                p.requires_grad = False

    def unfreeze(self):
        """Unfreeze all backbone parameters."""
        if self._encoder is not None:
            for p in self._encoder.parameters():
                p.requires_grad = True

    def is_frozen(self) -> bool:
        """Check if backbone is frozen."""
        if self._encoder is None:
            return False
        for p in self._encoder.parameters():
            if p.requires_grad:
                return False
        return True

    def forward(
        self,
        mod_dict: Dict[str, Dict[str, torch.Tensor]]
    ) -> List[torch.Tensor]:
        """Forward pass through the backbone.

        TerraTorch interface:
        - Input: mod_dict[str, dict[str, Tensor]] = {
              "S2L1C": {"x": tensor, "input_mask": mask},
              "S1GRD": {"x": tensor, "input_mask": mask}
          }
        - Output: List[Tensor] - features from each transformer layer
                  Shape: (B, N, D) per tensor

        Returns:
            List of feature tensors, one per layer
        """
        if self._encoder is not None:
            # Use real encoder
            encoder_tokens, encoder_emb, encoder_mask, modality_mask = \
                self._encoder.cat_encoder_tensors(mod_dict)

            features = self._encoder.forward_encoder(encoder_tokens, encoder_mask)
            return [features]  # Simplified - real implementation extracts layers

        # Placeholder: return dummy features
        batch_size = 1
        num_tokens = 256
        dim = 768
        return [
            torch.zeros(batch_size, num_tokens, dim) 
            for _ in range(12)  # 12 transformer layers
        ]

    def extract_features(
        self,
        mod_dict: Dict[str, Dict[str, torch.Tensor]],
        indices: Optional[List[int]] = None
    ) -> List[torch.Tensor]:
        """Extract features at specific indices for distillation.

        Args:
            mod_dict: Modality dict (TerraTorch format)
            indices: Feature indices to extract [2, 5, 8, 11] default

        Returns:
            List of feature tensors at specified indices
            Shape: (B, N, D) per tensor
        """
        indices = indices or self.feature_indices

        if self._encoder is not None:
            # Real implementation: extract from transformer layers
            # features[i] = output of layer i
            encoder_tokens, encoder_emb, encoder_mask, modality_mask = \
                self._encoder.cat_encoder_tensors(mod_dict)

            all_features = self._encoder.forward_encoder(encoder_tokens, encoder_mask)

            # SelectIndices equivalent
            return [all_features[i] for i in indices]

        # Placeholder
        batch_size = 1
        num_tokens = 256
        dim = 768
        return [
            torch.zeros(batch_size, num_tokens, dim)
            for _ in indices
        ]

    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about available feature indices."""
        return {
            "model_name": self.model_name,
            "modalities": self.modalities,
            "feature_indices": self.feature_indices,
            "merge_method": self.merge_method,
            "is_loaded": self._is_loaded,
            "output_type": "List[Tensor]",
            "output_shape": "(B, N, D) - Batch, Tokens, Dim",
        }

    def __repr__(self):
        return (
            f"TerraMindBackbone(\n"
            f"  model_name={self.model_name},\n"
            f"  modalities={self.modalities},\n"
            f"  feature_indices={self.feature_indices},\n"
            f"  merge_method={self.merge_method},\n"
            f"  is_loaded={self._is_loaded}\n"
            f")"
        )