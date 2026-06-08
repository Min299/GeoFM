"""geofm.models.peft.lora_targets

Target module names for LoRA injection.

Note: MLP targeting is disabled by default.
Geospatial PEFT literature shows attention LoRA captures most gains.
MLP LoRA doubles parameter count with marginal benefit.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LoRATargets:
    """Target module names for LoRA injection.

    These are the default target module names used in various vision transformers.
    Different models may use different naming conventions.

    Note: MLP targets (fc1, fc2, fc3) are disabled by default.
    Only attention targets (qkv, proj) are enabled.

    Attributes:
        qkv: Query, Key, Value projection module name
        proj: Output projection module name
        fc1: First MLP layer module name (disabled by default)
        fc2: Second MLP layer module name (disabled by default)
        fc3: Third MLP layer module name (disabled by default)
    """

    qkv: str = "attn.qkv"
    proj: str = "attn.proj"
    fc1: str = ""  # Disabled
    fc2: str = ""  # Disabled
    fc3: str = ""  # Disabled


# Preset target configurations for different backbones
# Note: MLP targets are disabled by default per geospatial PEFT literature
TERRAMIND_TARGETS = LoRATargets(
    qkv="attn.qkv",
    proj="attn.proj",
    fc1="",  # Disabled
    fc2="",  # Disabled
)

PRITHVI_TARGETS = LoRATargets(
    qkv="attention.qkv",
    proj="attention.proj",
    fc1="",  # Disabled
    fc2="",  # Disabled
)

VIT_TARGETS = LoRATargets(
    qkv="attn.qkv",
    proj="attn.proj",
    fc1="",  # Disabled
    fc2="",  # Disabled
    fc3="",  # Disabled
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
        include_mlp: Whether to include MLP layers (default: False)

    Returns:
        List of module name patterns (empty strings filtered out)
    """
    names = [targets.qkv, targets.proj]

    if include_mlp:
        if targets.fc1:
            names.append(targets.fc1)
        if targets.fc2:
            names.append(targets.fc2)
        if targets.fc3:
            names.append(targets.fc3)

    # Filter out empty strings
    return [n for n in names if n]