"""geofm.models.flood_model

First real runnable model: TerraMind + Decoder + Flood Segmentation.

Single task. Single dataset. Single objective.
No multitasking.

Architecture:
    TerraMind (backbone)
        ↓
    LoRA OR Full FT
        ↓
    UNet Decoder
        ↓
    Flood Mask
"""
from typing import Dict, Optional, List
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class FloodModelConfig:
    """Configuration for flood segmentation model."""

    backbone: str = "terramind_v1_base"
    num_classes: int = 2  # background, flood
    pretrained: bool = True
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    # Decoder
    decoder_channels: List[int] = (512, 256, 128, 64)
    decoder_use_batchnorm: bool = True

    # Feature extraction
    feature_indices: List[int] = None

    def __post_init__(self):
        if self.feature_indices is None:
            if "large" in self.backbone:
                self.feature_indices = [5, 11, 17, 23]
            else:
                self.feature_indices = [2, 5, 8, 11]


class FloodModel(nn.Module):
    """Flood segmentation model.

    Architecture:
        Input (mod_dict) -> TerraMind Backbone -> Decoder -> Flood Mask

    Usage:
        # Create model
        model = FloodModel(config=FloodModelConfig(backbone="terramind_v1_base"))

        # Forward pass
        mod_dict = {"S2L1C": {"x": s2_tensor}}
        output = model(mod_dict)  # (B, num_classes, H, W)

        # Training
        loss = criterion(output, target)
    """

    def __init__(
        self,
        config: Optional[FloodModelConfig] = None,
        **kwargs
    ):
        super().__init__()

        self.config = config or FloodModelConfig(**kwargs)
        self.num_classes = self.config.num_classes

        # Backbone
        from geofm.models.backbones.terramind_factory import create_terramind_config
        terramind_config = create_terramind_config(
            model_name=self.config.backbone,
            pretrained=self.config.pretrained,
        )
        self.backbone = self._create_backbone(terramind_config)

        # Apply LoRA if enabled
        if self.config.use_lora:
            from geofm.models.lora import inject_lora
            try:
                self.backbone = inject_lora(
                    self.backbone,
                    rank=self.config.lora_rank,
                    alpha=self.config.lora_alpha,
                    dropout=self.config.lora_dropout,
                )
                print(f"LoRA applied with rank={self.config.lora_rank}")
            except ValueError as e:
                # LoRA injection failed (model may not have standard attention modules)
                # Continue without LoRA for placeholder backbones
                print(f"Warning: LoRA injection failed ({e}). Continuing without LoRA.")
                self.config.use_lora = False

        # Feature extractor
        from geofm.models.backbones.feature_extractor import FeatureExtractor
        self.feature_extractor = FeatureExtractor(
            self.backbone,
            feature_indices=self.config.feature_indices
        )

        # Decoder
        self.decoder = self._create_decoder()

        # Segmentation head
        self.segmentation_head = nn.Conv2d(
            self.config.decoder_channels[-1],
            self.num_classes,
            kernel_size=1
        )

    def _create_backbone(self, terramind_config):
        """Create the backbone model.

        Note: In production, this would use TerraMindFactory.build().
        For now, uses placeholder that matches interface.
        """
        from geofm.models.backbones.terramind_backbone import TerraMindBackbone
        return TerraMindBackbone(
            model_name=terramind_config.model_name,
            pretrained=terramind_config.pretrained,
            modalities=terramind_config.modalities,
        )

    def _create_decoder(self):
        """Create UNet-style decoder."""
        channels = self.config.decoder_channels

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        in_channels = channels[0]  # From backbone

        for out_channels in channels[1:]:
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels) if self.config.decoder_use_batchnorm else nn.Identity(),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels) if self.config.decoder_use_batchnorm else nn.Identity(),
                    nn.ReLU(inplace=True),
                )
            )
            in_channels = out_channels

        # Upsampling
        self.upsample = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )

    def forward(
        self,
        mod_dict: Dict[str, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            mod_dict: Modality dict in TerraTorch format
                {"S2L1C": {"x": tensor, "input_mask": mask}}

        Returns:
            Segmentation logits (B, num_classes, H, W)
        """
        # Extract features
        features = self.feature_extractor(mod_dict)

        # Use last feature for simplicity
        # In full implementation, would use all features with skip connections
        x = features[-1]  # (B, N, D) where N=tokens, D=dim

        # Reshape: (B, N, D) -> (B, D, H, W)
        # Approximate: assume square spatial dimensions
        B, N, D = x.shape
        H = W = int(N ** 0.5)  # Approximate

        if H * W != N:
            # Not square, use fallback
            H = W = 16

        x = x.permute(0, 2, 1).view(B, D, H, W)

        # Decode
        for decoder_block in self.decoder_blocks:
            x = self.upsample(x)
            x = decoder_block(x)

        # Segmentation head
        output = self.segmentation_head(x)

        # Upsample to original size (placeholder - real impl would use actual size)
        output = nn.functional.interpolate(
            output,
            scale_factor=4,  # Approximate upsampling
            mode="bilinear",
            align_corners=False
        )

        return output

    def get_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_trainable_params(self) -> None:
        """Print trainable parameter count."""
        total = self.get_trainable_params()
        print(f"Trainable parameters: {total:,}")


def create_flood_model(
    backbone: str = "terramind_v1_base",
    use_lora: bool = False,
    lora_rank: int = 16,
    **kwargs
) -> FloodModel:
    """Create a flood segmentation model.

    Args:
        backbone: Backbone model name
        use_lora: Whether to use LoRA
        lora_rank: LoRA rank if using LoRA
        **kwargs: Additional config parameters

    Returns:
        FloodModel instance
    """
    config = FloodModelConfig(
        backbone=backbone,
        use_lora=use_lora,
        lora_rank=lora_rank,
        **kwargs
    )
    return FloodModel(config=config)