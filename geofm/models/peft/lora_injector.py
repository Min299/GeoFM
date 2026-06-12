"""geofm.models.peft.lora_injector

LoRA injection into transformer models.
Injects LoRA layers into attention modules (qkv, proj).
"""
from __future__ import annotations

from typing import List, Optional

import torch.nn as nn

from geofm.models.peft.lora_adapter import LoRALinear


class LoRAInjector:
    """Inject LoRA layers into transformer models.

    This injector targets attention layers specifically:
    - attn.qkv: Query, Key, Value projection
    - attn.proj: Output projection

    MLP targeting is disabled by default (per geospatial PEFT literature).

    Usage:
        injector = LoRAInjector(rank=16, alpha=32)
        replaced = injector.inject(model)
        print(f"Replaced {len(replaced)} layers")
    """

    def __init__(
        self,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
        target_qkv: bool = True,
        target_proj: bool = True,
    ):
        """Initialize LoRA injector.

        Args:
            rank: LoRA rank (default: 16)
            alpha: LoRA alpha scaling factor (default: 32)
            dropout: Dropout probability (default: 0.05)
            target_qkv: Whether to inject into QKV layers (default: True)
            target_proj: Whether to inject into projection layers (default: True)
        """
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_qkv = target_qkv
        self.target_proj = target_proj

    def inject(self, model: nn.Module) -> List[str]:
        """Inject LoRA layers into model.

        Args:
            model: Model to inject LoRA into

        Returns:
            List of module names that were replaced
        """
        replaced = []

        # Find all modules to replace
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            # Check if this is an attention target
            if self._is_attention_target(name):
                replaced.append(name)

        # Replace each module
        for name in replaced:
            self._replace(model, name)

        return replaced

    def _is_attention_target(self, name: str) -> bool:
        """Check if module name is an attention target.

        Args:
            name: Module name

        Returns:
            True if should be replaced
        """
        # Check qkv
        if self.target_qkv and name.endswith("attn.qkv"):
            return True

        # Check proj
        if self.target_proj and name.endswith("attn.proj"):
            return True

        return False

    def _replace(
        self,
        model: nn.Module,
        module_name: str,
    ) -> None:
        """Replace a module with LoRA version.

        Args:
            model: Model containing the module
            module_name: Full path to module
        """
        # Navigate to parent
        parts = module_name.split(".")
        parent = model

        for part in parts[:-1]:
            parent = getattr(parent, part)

        child_name = parts[-1]

        # Get old layer
        old_layer = getattr(parent, child_name)

        # Create LoRA version
        lora_layer = LoRALinear(
            base_layer=old_layer,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
        )

        # Replace
        setattr(parent, child_name, lora_layer)

    def freeze_except_lora(self, model: nn.Module) -> None:
        """Freeze all parameters except LoRA layers.

        Args:
            model: Model to freeze
        """
        # First freeze everything
        for p in model.parameters():
            p.requires_grad = False

        # Then unfreeze LoRA parameters
        for module in model.modules():
            if isinstance(module, LoRALinear):
                for p in module.parameters():
                    p.requires_grad = True

    def count_injected(self, model: nn.Module) -> int:
        """Count number of LoRA layers injected.

        Args:
            model: Model to check

        Returns:
            Number of LoRA layers
        """
        count = 0
        for module in model.modules():
            if isinstance(module, LoRALinear):
                count += 1
        return count

    def get_injected_names(self, model: nn.Module) -> List[str]:
        """Get names of all injected LoRA layers.

        Args:
            model: Model to check

        Returns:
            List of module names with LoRA
        """
        names = []
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear):
                names.append(name)
        return names


def inject_lora_into_backbone(
    backbone: nn.Module,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
) -> LoRAInjector:
    """Convenience function to inject LoRA into a backbone.

    Args:
        backbone: Backbone model
        rank: LoRA rank
        alpha: LoRA alpha
        dropout: Dropout probability

    Returns:
        LoRAInjector instance
    """
    injector = LoRAInjector(
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )
    injector.inject(backbone)
    injector.freeze_except_lora(backbone)
    return injector