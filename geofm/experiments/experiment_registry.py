"""geofm.experiments.experiment_registry

Registry of all adaptation experiments.

Each experiment defines:
- adapter: Which adapter to use (feature, lora, hybrid, None)
- freeze_backbone: Whether to freeze the backbone
"""
from __future__ import annotations


# Experiment registry
EXPERIMENTS = {
    "feature": {
        "name": "Feature Adapter",
        "adapter": "feature",
        "freeze_backbone": True,
        "description": "Bottleneck feature adapter (dim → bottleneck_dim → dim)",
    },
    "lora": {
        "name": "Low-Rank Adaptation",
        "adapter": "lora",
        "freeze_backbone": True,
        "rank": 16,
        "alpha": 32,
        "description": "LoRA with rank=16, alpha=32 on attention layers",
    },
    "hybrid": {
        "name": "Hybrid (Feature + LoRA)",
        "adapter": "hybrid",
        "freeze_backbone": True,
        "description": "Feature adapter + LoRA combined",
    },
    "fullft": {
        "name": "Full Fine-Tuning",
        "adapter": None,
        "freeze_backbone": False,
        "description": "All parameters trainable",
    },
}


def get_experiment(name: str) -> dict:
    """Get experiment config by name.

    Args:
        name: Experiment name (feature, lora, hybrid, fullft)

    Returns:
        Experiment configuration dict

    Raises:
        KeyError: If experiment not found
    """
    if name not in EXPERIMENTS:
        raise KeyError(f"Unknown experiment: {name}. Available: {list(EXPERIMENTS.keys())}")
    return EXPERIMENTS[name]


def list_experiments() -> list:
    """List all available experiments.

    Returns:
        List of experiment names
    """
    return list(EXPERIMENTS.keys())


def get_adapter_type(name: str) -> str:
    """Get adapter type for experiment.

    Args:
        name: Experiment name

    Returns:
        Adapter type string
    """
    return get_experiment(name)["adapter"]


def should_freeze_backbone(name: str) -> bool:
    """Check if backbone should be frozen for experiment.

    Args:
        name: Experiment name

    Returns:
        True if backbone should be frozen
    """
    return get_experiment(name)["freeze_backbone"]