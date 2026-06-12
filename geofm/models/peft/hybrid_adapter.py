"""geofm.models.peft.hybrid_adapter

Hybrid adapter combining LoRA with feature adapters.

This is the main adapter that combines:
1. LoRA for transformer attention layers
2. Feature adapters for multi-scale feature processing
"""
import torch
import torch.nn as nn

from .lora_adapter import TerraMindLoRA


class HybridAdapter(nn.Module):
    """Hybrid adapter combining LoRA and feature adapters.

    Architecture:
        Input -> Backbone (frozen) -> Features [F2, F5, F8, F11]
                                            |
                                            v
                                    Feature Adapter (trainable)
                                            |
                                            v
                                    Decoder (task-specific)
                                            |
                                            v
                                        Output

    Usage:
        # Build components
        backbone = build_backbone("terramind_base")
        feature_adapter = TaskFeatureAdapter([768, 768, 768, 768])
        decoder_bank = DecoderBank()

        # Create hybrid
        hybrid = HybridAdapter(
            backbone=backbone,
            lora_rank=16,  # Apply LoRA to attention
            feature_adapter=feature_adapter,
            decoder_bank=decoder_bank,
            task="flood",
        )

        # Forward pass
        output = hybrid(input_tensor)
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_adapter: nn.Module = None,
        decoder_bank: nn.Module = None,
        lora_rank: int = 16,
        lora_alpha: int = 16,
        task: str = "flood",
    ):
        super().__init__()

        self.backbone = backbone
        self.feature_adapter = feature_adapter
        self.decoder_bank = decoder_bank
        self.task = task

        # LoRA is applied after feature extraction
        self.lora_adapter = TerraMindLoRA(
            backbone,
            rank=lora_rank,
            alpha=lora_alpha,
        )

    def set_task(self, task: str):
        """Set the active task for multi-task learning.

        Args:
            task: Task name (e.g., "flood", "burn", "lulc")
        """
        self.task = task

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the hybrid adapter.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Task-specific output (e.g., segmentation mask)
        """
        # Extract features from backbone
        if hasattr(self.backbone, 'extract_features'):
            features = self.backbone.extract_features(x)
        else:
            features = self.backbone(x)

        # Apply feature adapter if present
        if self.feature_adapter is not None:
            features = self.feature_adapter(features)

        # Apply task-specific decoder if present
        if self.decoder_bank is not None:
            if hasattr(self.decoder_bank, 'forward'):
                return self.decoder_bank(features, self.task)

        return features

    def freeze_backbone(self):
        """Freeze the backbone parameters."""
        if hasattr(self.backbone, 'freeze'):
            self.backbone.freeze()

    def unfreeze_backbone(self):
        """Unfreeze the backbone parameters."""
        if hasattr(self.backbone, 'unfreeze'):
            self.backbone.unfreeze()

    def count_params(self) -> dict:
        """Count parameters by component.

        Returns:
            Dictionary with parameter counts
        """
        counts = {
            "backbone": 0,
            "lora": 0,
            "feature_adapter": 0,
            "decoder": 0,
            "total": 0,
        }

        for name, param in self.named_parameters():
            counts["total"] += param.numel()

            if "lora" in name:
                counts["lora"] += param.numel()
            elif "feature_adapter" in name:
                counts["feature_adapter"] += param.numel()
            elif "decoder" in name:
                counts["decoder"] += param.numel()

        return counts


class LoRAFeatureHybrid(nn.Module):
    """Simplified hybrid for experiments.

    Combines LoRA with a simple feature adapter for
    quick experimentation.
    """

    def __init__(
        self,
        backbone: nn.Module,
        lora_rank: int = 16,
        lora_alpha: int = 16,
    ):
        super().__init__()

        self.backbone = backbone

        # Apply LoRA
        self.lora = TerraMindLoRA(
            backbone,
            rank=lora_rank,
            alpha=lora_alpha,
        )

        # Simple feature adapter
        self.feature_proj = nn.ModuleDict()

    def add_task_adapter(
        self,
        task: str,
        in_channels: list,
        out_channels: int,
    ):
        """Add a task-specific adapter.

        Args:
            task: Task name
            in_channels: List of input channels per feature level
            out_channels: Number of output channels
        """
        self.feature_proj[task] = nn.Conv2d(
            in_channels[0],
            out_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor, task: str = "default") -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor
            task: Task name

        Returns:
            Output tensor
        """
        # Extract features
        features = self.backbone(x)

        # Get task adapter
        if task in self.feature_proj:
            # Apply task-specific projection
            # features is List[Tensor] (B, N, D)
            # Need to reshape to (B, D, H, W) for convolution
            feat = features[0]  # Use first feature level
            B, N, D = feat.shape
            H = W = int(N ** 0.5)
            feat = feat.view(B, H, W, D).permute(0, 3, 1, 2)  # (B, D, H, W)
            return self.feature_proj[task](feat)

        return features[0] if features else None