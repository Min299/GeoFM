"""geofm.experiments.experiment_builder

Factory for building experiment models.
"""
from typing import Optional
import torch
import torch.nn as nn

from geofm.models.backbones import build_backbone
from geofm.models.peft import (
    LoRALinearAdapter as TerraMindLoRA,
    TaskFeatureAdapter,
    HybridAdapter,
)
from .experiment import ExperimentConfig


class ExperimentBuilder:
    """Factory for building experiment models.

    Usage:
        config = ExperimentConfig(
            name="exp01",
            task="flood",
            adaptation="lora",
        )
        model = ExperimentBuilder.build(config)
    """

    @staticmethod
    def build(config: ExperimentConfig) -> nn.Module:
        """Build a model for the given experiment.

        Args:
            config: Experiment configuration

        Returns:
            Model ready for training
        """
        if config.adaptation == "feature":
            return ExperimentBuilder._build_feature_model(config)
        elif config.adaptation == "lora":
            return ExperimentBuilder._build_lora_model(config)
        elif config.adaptation == "hybrid":
            return ExperimentBuilder._build_hybrid_model(config)
        elif config.adaptation == "full_ft":
            return ExperimentBuilder._build_full_ft_model(config)
        else:
            raise ValueError(f"Unknown adaptation: {config.adaptation}")

    @staticmethod
    def _build_feature_model(config: ExperimentConfig) -> nn.Module:
        """Build model with feature adapter only.

        Args:
            config: Experiment configuration

        Returns:
            Feature adapter model
        """
        # Build backbone
        backbone = build_backbone(
            config.backbone,
            pretrained=True,
        )
        backbone.freeze()

        # Get feature dimensions from backbone
        feature_dim = getattr(backbone, "feature_dim", 768)
        feature_channels = [feature_dim] * 4  # 4 feature levels

        # Build feature adapter
        feature_adapter = TaskFeatureAdapter(
            in_channels=feature_channels,
            out_channels=1,  # Binary segmentation
        )

        # Wrap in simple adapter module
        model = FeatureAdapterModel(
            backbone=backbone,
            feature_adapter=feature_adapter,
        )

        return model

    @staticmethod
    def _build_lora_model(config: ExperimentConfig) -> nn.Module:
        """Build model with LoRA adaptation.

        Args:
            config: Experiment configuration

        Returns:
            LoRA-adapted model
        """
        # Build backbone
        backbone = build_backbone(
            config.backbone,
            pretrained=True,
        )
        backbone.freeze()

        # Apply LoRA
        lora_model = TerraMindLoRA(
            backbone,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            freeze_backbone=True,
        )

        return lora_model

    @staticmethod
    def _build_hybrid_model(config: ExperimentConfig) -> nn.Module:
        """Build model with hybrid (LoRA + Feature) adaptation.

        Args:
            config: Experiment configuration

        Returns:
            Hybrid model
        """
        # Build backbone
        backbone = build_backbone(
            config.backbone,
            pretrained=True,
        )
        backbone.freeze()

        # Get feature dimensions
        feature_dim = getattr(backbone, "feature_dim", 768)
        feature_channels = [feature_dim] * 4

        # Build feature adapter
        feature_adapter = TaskFeatureAdapter(
            in_channels=feature_channels,
            out_channels=1,
        )

        # Build LoRA model
        lora_backbone = TerraMindLoRA(
            backbone,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            freeze_backbone=True,
        )

        # Build hybrid model
        model = HybridAdapter(
            backbone=lora_backbone,
            feature_adapter=feature_adapter,
        )

        return model

    @staticmethod
    def _build_full_ft_model(config: ExperimentConfig) -> nn.Module:
        """Build model with full fine-tuning.

        Args:
            config: Experiment configuration

        Returns:
            Fully trainable model
        """
        backbone = build_backbone(
            config.backbone,
            pretrained=True,
        )
        return backbone


class FeatureAdapterModel(nn.Module):
    """Simple model combining backbone and feature adapter.

    This is the minimal model for feature-adapter experiments.
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_adapter: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.feature_adapter = feature_adapter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Output tensor
        """
        # Extract features
        if hasattr(self.backbone, "extract_features"):
            features = self.backbone.extract_features(x)
        else:
            features = self.backbone(x)

        # Apply feature adapter
        return self.feature_adapter(features)

    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_experiment_model(
    name: str,
    task: str,
    adaptation: str,
    backbone: str = "terramind_base",
    **kwargs,
) -> nn.Module:
    """Convenience function to build an experiment model.

    Args:
        name: Experiment name
        task: Task name
        adaptation: Adaptation method
        backbone: Backbone name
        **kwargs: Additional config options

    Returns:
        Model ready for training
    """
    config = ExperimentConfig(
        name=name,
        task=task,
        adaptation=adaptation,
        backbone=backbone,
        **kwargs,
    )
    return ExperimentBuilder.build(config)