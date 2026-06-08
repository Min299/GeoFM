"""geofm.models.peft.lora_config

Configuration for LoRA adapters.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation).

    Attributes:
        rank: LoRA rank (default: 16). Higher = more parameters, better quality.
        alpha: LoRA alpha scaling factor (default: 32). Usually set to 2x rank.
        dropout: Dropout probability for LoRA layers (default: 0.05).
        target_qkv: Whether to apply LoRA to QKV attention projections.
        target_proj: Whether to apply LoRA to output projection.
        target_mlp: Whether to apply LoRA to MLP layers.
    """

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_qkv: bool = True
    target_proj: bool = True
    target_mlp: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {self.rank}")

        if self.alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {self.alpha}")

        if not (0 <= self.dropout < 1):
            raise ValueError(f"LoRA dropout must be in [0, 1), got {self.dropout}")

        if self.alpha < self.rank:
            import warnings
            warnings.warn(
                f"LoRA alpha ({self.alpha}) < rank ({self.rank}). "
                "Consider setting alpha = 2 * rank for better performance."
            )

    @property
    def scale(self) -> float:
        """Get LoRA scaling factor.

        Returns:
            alpha / rank
        """
        return self.alpha / self.rank

    def get_rank_dict(self) -> dict:
        """Get rank configuration as dictionary.

        Returns:
            Dictionary with rank settings
        """
        return {
            "q": self.rank,
            "k": self.rank,
            "v": self.rank,
        }


# Preset configurations
LORA_CONFIGS = {
    "tiny": LoRAConfig(rank=4, alpha=8),
    "small": LoRAConfig(rank=8, alpha=16),
    "base": LoRAConfig(rank=16, alpha=32),
    "large": LoRAConfig(rank=32, alpha=64),
    "xlarge": LoRAConfig(rank=64, alpha=128),
}


def get_lora_config(name: str) -> LoRAConfig:
    """Get a preset LoRA configuration.

    Args:
        name: Configuration name (tiny, small, base, large, xlarge)

    Returns:
        LoRAConfig instance

    Raises:
        ValueError: If name is not a known preset
    """
    if name not in LORA_CONFIGS:
        raise ValueError(
            f"Unknown LoRA config: {name}. "
            f"Available: {list(LORA_CONFIGS.keys())}"
        )
    return LORA_CONFIGS[name]