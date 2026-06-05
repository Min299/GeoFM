"""geofm.models.lora.inject

LoRA injection utilities using PEFT.

Provides functions to inject LoRA adapters into models.
Target modules should be adjusted based on TerraMind internals.
"""
from typing import Optional

import torch.nn as nn

from geofm.models.lora.config import LoRAConfig


def inject_lora(
    model: nn.Module,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.1,
    target_modules: Optional[list] = None
) -> nn.Module:
    """Inject LoRA into a model using PEFT.

    Args:
        model: Model to inject LoRA into
        rank: LoRA rank
        alpha: LoRA alpha scaling
        dropout: Dropout probability
        target_modules: List of module names to apply LoRA to
            Default: ["q_proj", "k_proj", "v_proj"]

    Returns:
        Model with LoRA adapters
    """
    from peft import LoraConfig as PEFTLoraConfig, get_peft_model

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj"]

    config = PEFTLoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=target_modules
    )

    return get_peft_model(model, config)


def inject_lora_from_config(
    model: nn.Module,
    config: LoRAConfig
) -> nn.Module:
    """Inject LoRA using a LoRAConfig."""
    from peft import get_peft_model

    peft_config = config.to_peft_config()
    return get_peft_model(model, peft_config)


def inject_task_lora(
    model: nn.Module,
    task_name: str,
    rank: int = 16,
    alpha: int = 32
) -> nn.Module:
    """Inject task-specific LoRA."""
    return inject_lora(model, rank=rank, alpha=alpha)


def print_trainable_params(model: nn.Module) -> None:
    """Print number of trainable parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
