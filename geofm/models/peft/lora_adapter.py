"""geofm.models.peft.lora_adapter

LoRA adapter for TerraMind backbone.

Injects LoRA layers into the attention modules of a
transformer model using explicit encoder traversal.
"""
from typing import Optional
import torch.nn as nn

from .lora_layer import LoRALinear, LoRAConfig


def inject_lora_explicit(
    backbone,
    rank: int = 16,
    alpha: int = 16,
    target_qkv: bool = True,
    target_proj: bool = True,
) -> int:
    """Inject LoRA layers using explicit encoder traversal.

    This is the preferred method as it explicitly targets the
    transformer encoder blocks, avoiding ambiguity with generic
    module traversal.

    Args:
        backbone: The TerraMindBackbone wrapper
        rank: LoRA rank
        alpha: LoRA alpha (scaling factor)
        target_qkv: Whether to inject into QKV layers
        target_proj: Whether to inject into projection layers

    Returns:
        Number of layers replaced
    """
    replaced = 0

    # Get the inner TerraMindViT model
    inner = backbone._model if hasattr(backbone, '_model') else backbone

    # Explicitly traverse the encoder blocks
    if not hasattr(inner, 'encoder'):
        raise ValueError(f"Model {type(inner)} has no 'encoder' attribute")

    encoder = inner.encoder

    for block in encoder:
        if target_qkv and hasattr(block, 'attn') and hasattr(block.attn, 'qkv'):
            block.attn.qkv = LoRALinear(
                block.attn.qkv,
                rank=rank,
                alpha=alpha,
            )
            replaced += 1

        if target_proj and hasattr(block, 'attn') and hasattr(block.attn, 'proj'):
            block.attn.proj = LoRALinear(
                block.attn.proj,
                rank=rank,
                alpha=alpha,
            )
            replaced += 1

    return replaced


def freeze_all_except_lora(model):
    """Freeze all parameters except LoRA trainable parameters.

    Call this after LoRA injection to ensure only LoRA parameters
    are trainable.

    Args:
        model: Model with LoRA layers injected
    """
    # First freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Then unfreeze LoRA parameters
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True


class TerraMindLoRA(nn.Module):
    """LoRA adapter for TerraMind backbone.

    Injects LoRA layers into QKV and projection layers
    of all transformer blocks using explicit encoder traversal.

    Usage:
        # Load pretrained backbone
        backbone = build_backbone("terramind_base")

        # Apply LoRA with proper freezing
        lora_model = TerraMindLoRA(
            backbone,
            rank=16,
            alpha=16,
            freeze_backbone=True,  # Recommended: freeze original weights
        )

        # Now only LoRA parameters are trainable
        # Original weights remain frozen
    """

    def __init__(
        self,
        model: nn.Module,
        rank: int = 16,
        alpha: int = 16,
        target_qkv: bool = True,
        target_proj: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.model = model

        self.rank = rank
        self.alpha = alpha

        self.target_qkv = target_qkv
        self.target_proj = target_proj

        self._injected_layers = 0

        # Inject LoRA using explicit traversal
        self._injected_layers = inject_lora_explicit(
            self.model,
            rank=rank,
            alpha=alpha,
            target_qkv=target_qkv,
            target_proj=target_proj,
        )

        # Freeze backbone weights (only LoRA trainable)
        if freeze_backbone:
            freeze_all_except_lora(self.model)

    def inject(self):
        """Re-inject LoRA layers (clears and re-injects)."""
        # Note: This replaces the old injection logic
        # which used generic module traversal
        self._injected_layers = inject_lora_explicit(
            self.model,
            rank=self.rank,
            alpha=self.alpha,
            target_qkv=self.target_qkv,
            target_proj=self.target_proj,
        )

        # Re-freeze after re-injection
        freeze_all_except_lora(self.model)

    def forward(self, *args, **kwargs):
        """Forward pass through the model."""
        return self.model(*args, **kwargs)

    def count_lora_layers(self) -> int:
        """Count the number of LoRA layers injected."""
        count = 0
        for module in self.model.modules():
            if isinstance(module, LoRALinear):
                count += 1
        return count

    def count_lora_params(self) -> int:
        """Count total parameters in LoRA layers.

        Returns:
            Total number of LoRA parameters (lora_A + lora_B)
        """
        total = 0
        for module in self.model.modules():
            if isinstance(module, LoRALinear):
                total += module.lora_A.numel() + module.lora_B.numel()
        return total

    def merge_weights(self):
        """Merge LoRA weights into original layers."""
        for module in self.model.modules():
            if isinstance(module, LoRALinear):
                module.merge()

    def verify_frozen(self) -> bool:
        """Verify only LoRA parameters are trainable.

        Returns:
            True if only LoRA params are trainable
        """
        for name, param in self.model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                if not param.requires_grad:
                    return False
            else:
                # All other params should be frozen
                if param.requires_grad:
                    # Check if it's actually a LoRA param with different naming
                    if not any(x in name for x in ['lora_A', 'lora_B']):
                        return False
        return True

    def extra_repr(self) -> str:
        return (
            f"rank={self.rank}, alpha={self.alpha}, "
            f"injected_layers={self._injected_layers}, "
            f"lora_params={self.count_lora_params():,}"
        )


def inject_lora(
    model: nn.Module,
    config: LoRAConfig,
) -> TerraMindLoRA:
    """Inject LoRA into a model using a config.

    Args:
        model: The model to inject LoRA into
        config: LoRA configuration

    Returns:
        Model with LoRA injected
    """
    return TerraMindLoRA(
        model=model,
        rank=config.rank,
        alpha=config.alpha,
    )