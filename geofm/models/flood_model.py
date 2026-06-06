"""geofm.models.flood_model

First real runnable model: TerraMind + Decoder + Flood Segmentation.

Single task. Single dataset. Single objective.
No multitasking.

Architecture:
    TerraMind (backbone)
        ↓
    FeatureExtractor [2,5,8,11]
        ↓
    LoRA Adapters (if use_lora=True)
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


class LoRAAdapter(nn.Module):
    """LoRA adapter applied between feature extractor and decoder.

    Applied per feature level [f2, f5, f8, f11].
    """

    def __init__(self, dim: int = 768, rank: int = 16, alpha: int = 32):
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        # LoRA A: down projection
        self.lora_A = nn.Linear(dim, rank, bias=False)
        # LoRA B: up projection
        self.lora_B = nn.Linear(rank, dim, bias=False)
        # Scaling factor
        self.scaling = alpha / rank

        # Initialize A with Xavier, B with zeros
        nn.init.normal_(self.lora_A.weight, std=1 / rank)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        # x: (B, N, D)
        # LoRA: delta = B @ A @ x, output = x + scaling * delta
        return x + self.scaling * self.lora_B(self.lora_A(x))

    def get_trainable_params(self) -> int:
        """Get number of trainable parameters in this adapter."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TaskLoRAAdapter(nn.Module):
    """Task-specific LoRA adapter bank.

    Contains LoRA adapters for multiple tasks.
    Only one task's adapters are active at a time.

    Usage:
        # Create adapter bank for multiple tasks
        task_lora = TaskLoRAAdapter(tasks=["flood", "burn", "crop"], rank=16)

        # Activate task
        task_lora.set_task("flood")

        # Forward uses flood's adapters
        features = task_lora(features)
    """

    def __init__(
        self,
        tasks: List[str],
        dim: int = 768,
        rank: int = 16,
        alpha: int = 32,
    ):
        super().__init__()
        self.tasks = tasks
        self.dim = dim
        self.rank = rank
        self.alpha = alpha
        self._current_task = None

        # Create one LoRA adapter per feature level, per task
        # Structure: {task: [LoRA_f2, LoRA_f5, LoRA_f8, LoRA_f11]}
        self._task_adapters: Dict[str, nn.ModuleList] = {}

        for task in tasks:
            self._task_adapters[task] = nn.ModuleList([
                LoRAAdapter(dim=dim, rank=rank, alpha=alpha)
                for _ in range(4)  # 4 feature levels [2, 5, 8, 11]
            ])

    def set_task(self, task: str) -> None:
        """Activate a task's LoRA adapters.

        Args:
            task: Task name (must be in self.tasks)
        """
        if task not in self._task_adapters:
            raise ValueError(f"Unknown task: {task}. Available: {self.tasks}")
        self._current_task = task

    def get_task(self) -> Optional[str]:
        """Get current active task."""
        return self._current_task

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply task-specific LoRA to features.

        Args:
            features: List of feature tensors from backbone

        Returns:
            Features with LoRA adaptation applied
        """
        if self._current_task is None:
            return features

        adapters = self._task_adapters[self._current_task]
        return [adapter(feat) for adapter, feat in zip(adapters, features)]

    def register_task(self, task: str) -> None:
        """Register a new task with its LoRA adapters.

        Args:
            task: Task name
        """
        if task not in self._task_adapters:
            self._task_adapters[task] = nn.ModuleList([
                LoRAAdapter(dim=self.dim, rank=self.rank, alpha=self.alpha)
                for _ in range(4)
            ])
            self.tasks.append(task)

    def get_adapter_params(self, task: str) -> int:
        """Get number of trainable params for a task's adapters."""
        if task not in self._task_adapters:
            return 0
        return sum(
            p.numel()
            for adapter in self._task_adapters[task]
            for p in adapter.parameters()
            if p.requires_grad
        )


class FloodModel(nn.Module):
    """Flood segmentation model.

    Architecture:
        Input (mod_dict) -> TerraMind -> FeatureExtractor -> [TaskLoRAAdapter] -> Decoder -> Flood Mask

    Usage:
        # Single task model
        model = FloodModel(config=FloodModelConfig(backbone="terramind_v1_base"))
        output = model(mod_dict)

        # Multi-task model
        model = FloodModel(
            tasks=["flood", "burn"],  # Multi-task mode
            backbone="terramind_v1_base"
        )
        model.set_task("flood")  # Activate flood adapters
        output = model(mod_dict)
    """

    def __init__(
        self,
        config: Optional[FloodModelConfig] = None,
        tasks: Optional[List[str]] = None,  # For multi-task mode
        **kwargs
    ):
        super().__init__()

        self.config = config or FloodModelConfig(**kwargs)
        self.num_classes = self.config.num_classes
        self.is_multitask = tasks is not None
        self.tasks = tasks or [self.config.task if hasattr(self.config, 'task') else "default"]

        # Backbone
        from geofm.models.backbones.terramind_factory import create_terramind_config
        terramind_config = create_terramind_config(
            model_name=self.config.backbone,
            pretrained=self.config.pretrained,
        )
        self.backbone = self._create_backbone(terramind_config)

        # Feature extractor
        from geofm.models.features.feature_extractor import FeatureExtractor
        self.feature_extractor = FeatureExtractor(
            self.backbone,
            self.config.feature_indices
        )

        # Task-specific LoRA adapters between feature extractor and decoder
        if self.config.use_lora or self.is_multitask:
            self.task_lora = TaskLoRAAdapter(
                tasks=self.tasks,
                dim=768,
                rank=self.config.lora_rank,
                alpha=self.config.lora_alpha,
            )
            # Set default task
            self.task_lora.set_task(self.tasks[0])
            print(f"TaskLoRA adapters: {len(self.tasks)} tasks, rank={self.config.lora_rank}")
        else:
            self.task_lora = None

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

        # Decoder blocks - first one gets input from backbone (768 dim)
        self.decoder_blocks = nn.ModuleList()
        
        # Input projection: backbone features (768) -> decoder_channels[0]
        self.input_proj = nn.Sequential(
            nn.Conv2d(768, channels[0], kernel_size=1),
            nn.BatchNorm2d(channels[0]) if self.config.decoder_use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        
        in_channels = channels[0]  # After projection

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
        
        # Handle FeatureLevels or list
        if hasattr(features, 'to_list'):
            features = features.to_list()

        # Apply task-specific LoRA adapters
        if self.task_lora is not None:
            features = self.task_lora(features)

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

        # Project from backbone dim (768) to decoder channels[0]
        x = self.input_proj(x)

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

    def set_task(self, task: str) -> None:
        """Switch to a different task's LoRA adapters.

        Args:
            task: Task name
        """
        if self.task_lora is not None:
            self.task_lora.set_task(task)
            print(f"Switched to task: {task}")

    def get_current_task(self) -> Optional[str]:
        """Get the current active task."""
        if self.task_lora is not None:
            return self.task_lora.get_task()
        return None

    def get_task_params(self, task: str) -> int:
        """Get trainable params for a specific task's LoRA adapters."""
        if self.task_lora is not None:
            return self.task_lora.get_adapter_params(task)
        return 0

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