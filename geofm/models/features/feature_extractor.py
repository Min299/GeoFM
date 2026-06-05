"""geofm.models.features.feature_extractor

Feature extraction utilities for distillation and multi-task learning.
Extracts features at specific indices [2, 5, 8, 11] from TerraMind.

Key API (from TerraTorch inspection):
- Input: mod_dict[str, dict[str, Tensor]] = {
      "S2L1C": {"x": tensor, "input_mask": mask},
      "S1GRD": {"x": tensor, "input_mask": mask}
  }
- Output: List[Tensor] - one per transformer layer
- Shape: (B, N, D) where B=batch, N=tokens, D=dim
- Use SelectIndices neck to extract [2, 5, 8, 11]
"""
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class FeatureLevels:
    """Container for extracted feature levels.

    Attributes:
        f2, f5, f8, f11: Feature tensors at respective indices
        All features are normalized and ready for distillation loss
        Shape: (B, N, D) - Batch, Tokens, Dimension
    """
    f2: Optional[torch.Tensor] = None
    f5: Optional[torch.Tensor] = None
    f8: Optional[torch.Tensor] = None
    f11: Optional[torch.Tensor] = None

    # For large models: f5, f11, f17, f23
    f17: Optional[torch.Tensor] = None
    f23: Optional[torch.Tensor] = None

    def to_list(self) -> List[torch.Tensor]:
        """Convert to list, excluding None values."""
        return [f for f in [
            self.f2, self.f5, self.f8, self.f11, self.f17, self.f23
        ] if f is not None]

    def __iter__(self):
        return iter(self.to_list())

    @classmethod
    def from_list(cls, features: List[torch.Tensor], indices: List[int]) -> "FeatureLevels":
        """Create FeatureLevels from a list of features.

        Args:
            features: List of feature tensors
            indices: Corresponding indices [2, 5, 8, 11]

        Returns:
            FeatureLevels instance
        """
        result = cls()
        for feat, idx in zip(features, indices):
            setattr(result, f"f{idx}", feat)
        return result


class FeatureExtractor:
    """Extract intermediate features from TerraMind backbone.

    Based on TerraTorch's SelectIndices neck which exposes layers:
    [2, 5, 8, 11] for base/tiny/small models
    [5, 11, 17, 23] for large models

    Usage:
        extractor = FeatureExtractor(backbone, indices=[2, 5, 8, 11])

        # Extract features from sample
        features = extractor(modalities)

        # Use for distillation
        loss = mse_loss(student_features.f2, teacher_features.f2) + ...
    """

    # Default feature indices for each model variant
    DEFAULT_INDICES = {
        "terramind_v1_tiny": [2, 5, 8, 11],
        "terramind_v1_small": [2, 5, 8, 11],
        "terramind_v1_base": [2, 5, 8, 11],
        "terramind_v1_base_tim": [2, 5, 8, 11],
        "terramind_v1_large": [5, 11, 17, 23],
    }

    def __init__(
        self,
        backbone: nn.Module,
        indices: Optional[List[int]] = None,
    ):
        """Initialize feature extractor.

        Args:
            backbone: TerraMind backbone model
            indices: Feature indices to extract (default: [2, 5, 8, 11])
        """
        self.backbone = backbone
        self.indices = indices or [2, 5, 8, 11]

    def __call__(
        self,
        modalities: Dict[str, torch.Tensor]
    ) -> FeatureLevels:
        """Extract features from modalities dict.

        Args:
            modalities: Dict of modality tensors

        Returns:
            FeatureLevels object with f2, f5, f8, f11 tensors
        """
        features = self._extract_internal(modalities)

        result = FeatureLevels()
        for i, idx in enumerate(self.indices):
            if i < len(features):
                setattr(result, f"f{idx}", features[i])

        return result

    def _extract_internal(
        self,
        modalities: Dict[str, torch.Tensor]
    ) -> List[torch.Tensor]:
        """Internal feature extraction.

        This method extracts intermediate features from the backbone.
        The actual implementation depends on how TerraTorch exposes
        intermediate layer outputs.

        Returns:
            List of feature tensors at specified indices
        """
        # Try to get intermediate features from backbone
        if hasattr(self.backbone, 'extract_features'):
            return self.backbone.extract_features(modalities, self.indices)

        # Fallback: use forward pass
        output = self.backbone(modalities)
        features = output.get("features", output)

        # If single tensor, repeat for each index (placeholder)
        if isinstance(features, torch.Tensor):
            return [features] * len(self.indices)

        return list(features) if isinstance(features, (list, tuple)) else [features]

    def extract_for_distillation(
        self,
        teacher_modalities: Dict[str, torch.Tensor],
        student_modalities: Dict[str, torch.Tensor],
    ) -> Tuple[FeatureLevels, FeatureLevels]:
        """Extract features for distillation loss computation.

        Args:
            teacher_modalities: Teacher model input
            student_modalities: Student model input

        Returns:
            Tuple of (teacher_features, student_features)
        """
        teacher_features = self(teacher_modalities)
        student_features = self(student_modalities)

        return teacher_features, student_features


class DistillationLoss(nn.Module):
    """Feature matching distillation loss.

    Computes MSE loss between teacher and student features at
    specific indices [2, 5, 8, 11].

    Loss = MSE(F2_teacher, F2_student) +
           MSE(F5_teacher, F5_student) +
           MSE(F8_teacher, F8_student) +
           MSE(F11_teacher, F11_student)
    """

    def __init__(
        self,
        indices: Optional[List[int]] = None,
        weights: Optional[Dict[int, float]] = None,
    ):
        """Initialize distillation loss.

        Args:
            indices: Feature indices to match (default: [2, 5, 8, 11])
            weights: Optional per-index weights
        """
        super().__init__()
        self.indices = indices or [2, 5, 8, 11]
        self.weights = weights or {i: 1.0 for i in self.indices}
        self.mse = nn.MSELoss()

    def forward(
        self,
        teacher_features: FeatureLevels,
        student_features: FeatureLevels,
    ) -> torch.Tensor:
        """Compute distillation loss.

        Args:
            teacher_features: Teacher model features
            student_features: Student model features

        Returns:
            Scalar loss tensor
        """
        loss = 0.0

        for idx in self.indices:
            teacher_f = getattr(teacher_features, f"f{idx}", None)
            student_f = getattr(student_features, f"f{idx}", None)

            if teacher_f is not None and student_f is not None:
                # Align dimensions if needed
                if teacher_f.shape != student_f.shape:
                    student_f = self._align_features(student_f, teacher_f.shape)

                loss += self.weights[idx] * self.mse(teacher_f, student_f)

        return loss

    def _align_features(
        self,
        student_features: torch.Tensor,
        target_shape: torch.Size
    ) -> torch.Tensor:
        """Align student features to match teacher dimensions.

        Handles token sequence dimension (B, N, D) -> (B, N, D)
        """
        # Handle dimension mismatch with linear projection
        if student_features.shape[-1] != target_shape[-1]:
            proj = nn.Linear(student_features.shape[-1], target_shape[-1]).to(
                student_features.device
            )
            student_features = proj(student_features)

        # Handle token count mismatch with interpolation
        if student_features.shape[1] != target_shape[1]:
            student_features = torch.nn.functional.interpolate(
                student_features.permute(0, 2, 1),  # (B, D, N)
                size=target_shape[1],
                mode="linear",
                align_corners=False
            ).permute(0, 2, 1)  # Back to (B, N, D)

        return student_features


def create_feature_extractor(
    backbone: nn.Module,
    model_name: str = "terramind_v1_base",
) -> FeatureExtractor:
    """Create a feature extractor for the given backbone.

    Args:
        backbone: TerraMind backbone model
        model_name: Model variant to determine correct indices

    Returns:
        Configured FeatureExtractor
    """
    indices = FeatureExtractor.DEFAULT_INDICES.get(
        model_name,
        [2, 5, 8, 11]
    )
    return FeatureExtractor(backbone, indices=indices)