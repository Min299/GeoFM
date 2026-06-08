"""geofm.models.peft.lora_targets

Target module names for LoRA injection.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LoRATargets:
    """Target module names for LoRA injection.

    These are the default target module names used in various vision transformers.
    Different models may use different naming conventions.

    Attributes:
        qkv: Query, Key, Value projection module name
        proj: Output projection module name
        fc1: First MLP layer module name
        fc2: Second MLP layer module name
        fc3: Third MLP layer module name (if applicable)
    """

    qkv: str = "attn.qkv"
    proj: str = "attn.proj"
    fc1: str = "mlp.fc1"
    fc2: str = "mlp.fc2"
    fc3: str = "mlp.fc3"


# Preset target configurations for different backbones
TERRAMIND_TARGETS = LoRATargets(
    qkv="attn.qkv",
    proj="attn.proj",
    fc1="mlp.fc1",
    fc2="mlp.fc2",
)

PRITHVI_TARGETS = LoRATargets(
    qkv="attention.qkv",
    proj="attention.proj",
    fc1="mlp.fc1",
    fc2="mlp.fc2",
)

VIT_TARGETS = LoRATargets(
    qkv="attn.qkv",
    proj="attn.proj",
    fc1="mlp.fc1",
    fc2="mlp.fc2",
    fc3="mlp.fc3",
)

DEFAULT_LORA_TARGETS = TERRAMIND_TARGETS


def get_lora_targets(model_name: str) -> LoRATargets:
    """Get LoRA targets for a specific model.

    Args:
        model_name: Name of the model (terramind, prithvi, vit)

    Returns:
        LoRATargets instance

    Raises:
        ValueError: If model_name is not recognized
    """
    targets_map = {
        "terramind": TERRAMIND_TARGETS,
        "prithvi": PRITHVI_TARGETS,
        "vit": VIT_TARGETS,
        "terramind_base": TERRAMIND_TARGETS,
        "terramind_tiny": TERRAMIND_TARGETS,
        "terramind_small": TERRAMIND_TARGETS,
    }

    return targets_map.get(model_name.lower(), DEFAULT_LORA_TARGETS)


def get_target_names(targets: LoRATargets, include_mlp: bool = False) -> list:
    """Get list of target module names.

    Args:
        targets: LoRATargets instance
        include_mlp: Whether to include MLP layers

    Returns:
        List of module name patterns
    """
    names = [targets.qkv, targets.proj]

    if include_mlp:
        names.extend([targets.fc1, targets.fc2])
        if targets.fc3:
            names.append(targets.fc3)

    return names