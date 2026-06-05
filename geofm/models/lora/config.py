"""geofm.models.lora.config

Centralized LoRA configuration.

Controls rank, alpha, dropout, and target modules
for all LoRA adapters in the project.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapters.

    Attributes:
        rank: LoRA rank (default: 16)
            Higher = more parameters, potentially better quality
        alpha: LoRA alpha scaling (default: 32)
            Usually set to 2x rank
        dropout: Dropout probability (default: 0.1)
        target_modules: Which layers to apply LoRA to
        bias: Bias handling ("none", "all", "lora_only")
    """

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj"
    ])
    bias: str = "none"

    def to_peft_config(self):
        """Convert to PEFT LoraConfig."""
        from peft import LoraConfig as PEFTLoraConfig

        return PEFTLoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            bias=self.bias,
            target_modules=self.target_modules
        )


# Pre-defined configs for common scenarios
LORA_CONFIGS = {
    "light": LoRAConfig(rank=4, alpha=8, dropout=0.05),
    "standard": LoRAConfig(rank=16, alpha=32, dropout=0.1),
    "heavy": LoRAConfig(rank=64, alpha=128, dropout=0.1),
    "distillation": LoRAConfig(rank=8, alpha=16, dropout=0.0),
}