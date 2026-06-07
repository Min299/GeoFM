"""geofm.models.decoders.pyramid_decoder

Multi-scale pyramid decoder using all TerraMind feature levels.

Uses F2, F5, F8, F11 for segmentation.

Architecture:
    [F2, F5, F8, F11]
          ↓
    Feature Reshaping
    (B, N, D) → (B, D, H, W)
          ↓
    Multi-Scale Fusion
    (concat all 4 levels)
          ↓
    Decoder Blocks
    (progressive upsampling)
          ↓
    Segmentation Head
          ↓
    (B, num_classes, H, W)
"""
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureReshaper(nn.Module):
    """Convert (B, N, D) tokens to (B, D, H, W) spatial.

    Handles CLS token by removing it.
    Assumes N = 256 (16×16 spatial tokens + CLS).
    """

    def __init__(self, feature_dim: int = 768, spatial_size: int = 16):
        super().__init__()
        self.feature_dim = feature_dim
        self.spatial_size = spatial_size
        self.num_tokens = spatial_size * spatial_size  # 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) where N = 256 tokens + 1 CLS

        Returns:
            (B, D, H, W) where H=W=16
        """
        B, N, D = x.shape

        # Remove CLS token (index 0)
        if N > self.num_tokens:
            x = x[:, 1:, :]  # (B, 256, D)

        # Take exactly num_tokens
        x = x[:, :self.num_tokens, :]  # (B, 256, D)

        # Reshape to spatial: (B, D, 16, 16)
        x = x.view(B, D, self.spatial_size, self.spatial_size)

        return x


class MultiScaleFusion(nn.Module):
    """Fuse multi-scale features from all 4 levels.

    Simply concatenates all feature levels then projects to single representation.
    """

    def __init__(self, feature_dim: int = 768, num_levels: int = 4, fusion_channels: int = 512):
        super().__init__()
        self.num_levels = num_levels
        self.feature_dim = feature_dim

        # Each level: Conv2d(feature_dim → fusion_channels)
        self.level_convs = nn.ModuleList([
            nn.Conv2d(feature_dim, fusion_channels, kernel_size=1)
            for _ in range(num_levels)
        ])

        # Fusion: Conv2d(fusion_channels * num_levels → fusion_channels)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_channels * num_levels, fusion_channels, kernel_size=1),
            nn.BatchNorm2d(fusion_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of 4 tensors [F2, F5, F8, F11]
                     Each shape: (B, N, D) = (B, 256, 768)

        Returns:
            Fused features: (B, fusion_channels, 16, 16)
        """
        # Reshape and project each level
        projected = []
        for feat, conv in zip(features, self.level_convs):
            # (B, N, D) → (B, D, 16, 16)
            B, N, D = feat.shape
            spatial_size = 16

            # Handle CLS token removal: take N-1 tokens if N > spatial_size^2
            if N > spatial_size * spatial_size:
                feat = feat[:, 1:, :]  # Remove CLS token

            # Take exactly spatial_size^2 tokens
            num_tokens = spatial_size * spatial_size
            feat = feat[:, :num_tokens, :]  # (B, 256, D) or (B, 255, D)

            # If 255 tokens (after CLS removal), pad to 256
            if feat.shape[1] < num_tokens:
                pad_size = num_tokens - feat.shape[1]
                feat = F.pad(feat, (0, 0, 0, pad_size))  # Pad sequence dimension

            feat_spatial = feat.view(B, D, spatial_size, spatial_size)

            # Project
            proj = conv(feat_spatial)  # (B, fusion_channels, 16, 16)
            projected.append(proj)

        # Concatenate all levels
        fused = torch.cat(projected, dim=1)  # (B, fusion_channels * 4, 16, 16)

        # Fuse
        fused = self.fusion_conv(fused)  # (B, fusion_channels, 16, 16)

        return fused


class DecoderBlock(nn.Module):
    """UNet-style decoder block with upsampling and skip connections."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor
            skip: Skip connection tensor (optional)

        Returns:
            Decoded tensor
        """
        # Upsample
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        # Concatenate skip if provided
        if skip is not None:
            # Align spatial dimensions
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)

        # Conv blocks
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        return x


class PyramidDecoder(nn.Module):
    """Multi-scale pyramid decoder.

    Uses all 4 TerraMind feature levels [F2, F5, F8, F11].

    Architecture:
        [F2, F5, F8, F11] → Reshape → MultiScaleFusion → DecoderBlocks → SegHead

    Parameters:
        feature_dim: Dimension of input features (768 for base)
        num_classes: Number of output classes
        fusion_channels: Channels after multi-scale fusion
        decoder_channels: Channels in decoder blocks
    """

    def __init__(
        self,
        feature_dim: int = 768,
        num_classes: int = 2,
        fusion_channels: int = 512,
        decoder_channels: Tuple[int, ...] = (256, 128, 64),
    ):
        super().__init__()

        # Multi-scale feature fusion
        self.fusion = MultiScaleFusion(
            feature_dim=feature_dim,
            num_levels=4,
            fusion_channels=fusion_channels,
        )

        # Decoder blocks - 4 blocks to go from 16×16 → 128×128
        # Each block upsamples by 2×
        # 16 → 32 → 64 → 128 (3 blocks)
        self.decoder_blocks = nn.ModuleList()
        in_channels = fusion_channels

        for out_channels in decoder_channels:
            self.decoder_blocks.append(
                DecoderBlock(in_channels, 0, out_channels)  # 0 skip channels for first
            )
            in_channels = out_channels

        # Segmentation head
        self.seg_head = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)

        # Final upsample from 128×128 → 256×256 (2×)
        self.final_upsample = nn.Upsample(
            scale_factor=2,
            mode='bilinear',
            align_corners=False
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of 4 tensors [F2, F5, F8, F11]
                     Each shape: (B, N, D) = (B, 256, 768)

        Returns:
            Segmentation logits: (B, num_classes, 256, 256)
        """
        # Multi-scale fusion
        x = self.fusion(features)  # (B, fusion_channels, 16, 16)

        # Decode with progressive upsampling
        # 16 → 32 → 64 → 128 (3 blocks)
        for block in self.decoder_blocks:
            x = block(x)  # Upsample + conv

        # Segmentation
        logits = self.seg_head(x)  # (B, num_classes, 128, 128)

        # Final upsample to 256×256
        logits = self.final_upsample(logits)  # (B, num_classes, 256, 256)

        return logits


class PyramidDecoderWithSkips(nn.Module):
    """Pyramid decoder with skip connections from each feature level.

    More complex but potentially better for fine boundary prediction.
    """

    def __init__(
        self,
        feature_dim: int = 768,
        num_classes: int = 2,
        level_channels: Tuple[int, ...] = (256, 256, 256, 256),  # One per level
        decoder_channels: Tuple[int, ...] = (512, 256, 128, 64),
    ):
        super().__init__()
        self.num_levels = 4

        # Project each feature level to its channel dimension
        self.level_projs = nn.ModuleList([
            nn.Conv2d(feature_dim, level_channels[i], kernel_size=1)
            for i in range(self.num_levels)
        ])

        # Decoder blocks (deepest to shallowest)
        # Level 3 (F11) → Level 0 (F2)
        self.decoder_blocks = nn.ModuleList()

        # First block: deepest level only
        self.decoder_blocks.append(
            DecoderBlock(level_channels[3], 0, decoder_channels[0])
        )

        # Subsequent blocks: upsampled + skip from corresponding level
        for i in range(3):  # 0, 1, 2
            in_ch = decoder_channels[i]
            skip_ch = level_channels[2 - i]  # F8, F5, F2
            out_ch = decoder_channels[i + 1]
            self.decoder_blocks.append(
                DecoderBlock(in_ch, skip_ch, out_ch)
            )

        # Segmentation head
        self.seg_head = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)

        # Final upsample
        self.final_upsample = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of 4 tensors [F2, F5, F8, F11]
                     Each shape: (B, N, D) = (B, 256, 768)

        Returns:
            Segmentation logits: (B, num_classes, H, W)
        """
        # Reshape and project all levels
        spatial_features = []
        for feat, proj in zip(features, self.level_projs):
            B, N, D = feat.shape
            spatial_size = 16

            # Remove CLS token if present
            if N > spatial_size * spatial_size:
                feat = feat[:, 1:, :]

            # Take exactly spatial_size^2 tokens
            num_tokens = spatial_size * spatial_size
            feat = feat[:, :num_tokens, :]

            # Pad if needed
            if feat.shape[1] < num_tokens:
                pad_size = num_tokens - feat.shape[1]
                feat = F.pad(feat, (0, 0, 0, pad_size))

            feat_spatial = feat.view(B, D, spatial_size, spatial_size)
            spatial_features.append(proj(feat_spatial))

        # Decode: start with deepest (F11), add skips progressively
        # spatial_features: [F2, F5, F8, F11] = [0, 1, 2, 3]
        x = self.decoder_blocks[0](spatial_features[3])  # F11 only

        x = self.decoder_blocks[1](x, spatial_features[2])  # + F8
        x = self.decoder_blocks[2](x, spatial_features[1])  # + F5
        x = self.decoder_blocks[3](x, spatial_features[0])  # + F2

        # Segment
        logits = self.seg_head(x)
        logits = self.final_upsample(logits)

        return logits


def create_pyramid_decoder(
    name: str = "pyramid",
    feature_dim: int = 768,
    num_classes: int = 2,
    **kwargs
) -> nn.Module:
    """Factory function to create pyramid decoder.

    Args:
        name: Decoder variant ("pyramid" or "pyramid_with_skips")
        feature_dim: Input feature dimension
        num_classes: Number of output classes
        **kwargs: Additional arguments

    Returns:
        Pyramid decoder module
    """
    if name == "pyramid":
        return PyramidDecoder(feature_dim=feature_dim, num_classes=num_classes, **kwargs)
    elif name == "pyramid_with_skips":
        return PyramidDecoderWithSkips(feature_dim=feature_dim, num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown decoder: {name}")