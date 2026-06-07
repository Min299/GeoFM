"""geofm.models.peft.lora_adapter

LoRA adapter for TerraMind backbone.

Injects LoRA layers into the attention modules of a
transformer model.
"""
import torch.nn as nn

from .lora_layer import LoRALinear, LoRAConfig


class TerraMindLoRA(nn.Module):
    """LoRA adapter for TerraMind backbone.

    Injects LoRA layers into QKV and projection layers
    of all transformer blocks.

    Usage:
        # Load pretrained backbone
        backbone = build_backbone("terramind_base")

        # Apply LoRA
        lora_model = TerraMindLoRA(
            backbone,
            rank=16,
            alpha=16,
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
    ):
        super().__init__()

        self.model = model

        self.rank = rank
        self.alpha = alpha

        self.target_qkv = target_qkv
        self.target_proj = target_proj

        self._injected_layers = 0

        self.inject()

    def inject(self):
        """Inject LoRA layers into the model."""
        self._injected_layers = 0

        for name, module in self.model.named_modules():

            # Find attention modules
            if hasattr(module, "attn"):

                attn = module.attn

                # Inject into QKV layer
                if self.target_qkv and hasattr(attn, "qkv"):
                    if not isinstance(attn.qkv, LoRALinear):
                        attn.qkv = LoRALinear(
                            attn.qkv,
                            rank=self.rank,
                            alpha=self.alpha,
                        )
                        self._injected_layers += 1

                # Inject into projection layer
                if self.target_proj and hasattr(attn, "proj"):
                    if not isinstance(attn.proj, LoRALinear):
                        attn.proj = LoRALinear(
                            attn.proj,
                            rank=self.rank,
                            alpha=self.alpha,
                        )
                        self._injected_layers += 1

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
        """Count total parameters in LoRA layers."""
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