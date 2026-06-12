"""geofm.models.peft.lora_layer

LoRA (Low-Rank Adaptation) layer implementation.

Implements the LoRA technique from:
"LoRA: Low-Rank Adaptation of Large Language Models"
https://arxiv.org/abs/2106.09685

The key idea is to freeze the original weights and add
trainable low-rank decomposition matrices.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """LoRA wrapper around an existing Linear layer.

    W(x) + scale * B(A(x))

    Where:
    - W: frozen original weight
    - A: down-projection matrix (rank x in_features)
    - B: up-projection matrix (out_features x rank)
    - scale = alpha / rank

    Args:
        linear: The original nn.Linear layer to wrap
        rank: Rank of the low-rank matrices (default: 16)
        alpha: Scaling factor (default: 16)
    """

    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 16,
        alpha: int = 16,
    ):
        super().__init__()

        self.linear = linear

        self.rank = rank
        self.alpha = alpha

        self.scale = alpha / rank

        in_features = linear.in_features
        out_features = linear.out_features

        # LoRA trainable matrices
        self.lora_A = nn.Parameter(
            torch.zeros(rank, in_features)
        )

        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank)
        )

        # Initialize A with kaiming uniform (as per LoRA paper)
        nn.init.kaiming_uniform_(
            self.lora_A,
            a=math.sqrt(5),
        )

        # Initialize B with zeros (as per LoRA paper)
        nn.init.zeros_(
            self.lora_B
        )

        # Freeze the original layer
        self.linear.weight.requires_grad = False

        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation.

        Args:
            x: Input tensor

        Returns:
            Output tensor with LoRA delta applied
        """
        # Base output from frozen weights
        base = self.linear(x)

        # LoRA delta: scale * B @ A @ x
        # Using two linear calls instead of one to match
        # the original LoRA implementation behavior
        delta = (
            F.linear(
                F.linear(
                    x,
                    self.lora_A,
                ),
                self.lora_B,
            )
            * self.scale
        )

        return base + delta

    def merge(self):
        """Merge LoRA weights into the original layer.

        After calling this, the layer behaves like a regular
        linear layer with the adapted weights.
        """
        # W_merged = W + scale * B @ A
        merged_weight = self.linear.weight + self.scale * (self.lora_B @ self.lora_A)
        self.linear.weight.data = merged_weight

        # Make the merged weight the only trainable parameter
        self.linear.weight.requires_grad = True
        self.lora_A.requires_grad = False
        self.lora_B.requires_grad = False

    def extra_repr(self) -> str:
        return (
            f"in_features={self.linear.in_features}, "
            f"out_features={self.linear.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, scale={self.scale:.4f}"
        )


class LoRAConfig:
    """Configuration for LoRA adaptation.

    Args:
        rank: Rank of the low-rank matrices
        alpha: Scaling factor (typically same as rank)
        target_modules: List of module names to apply LoRA to
        dropout: Dropout probability (default: 0.0)
    """

    def __init__(
        self,
        rank: int = 16,
        alpha: int = 16,
        target_modules: list = None,
        dropout: float = 0.0,
    ):
        self.rank = rank
        self.alpha = alpha
        self.target_modules = target_modules or ["qkv", "proj"]
        self.dropout = dropout

    def __repr__(self):
        return (
            f"LoRAConfig(rank={self.rank}, alpha={self.alpha}, "
            f"target_modules={self.target_modules}, dropout={self.dropout})"
        )


def apply_lora_to_linear(
    linear: nn.Linear,
    config: LoRAConfig,
) -> LoRALinear:
    """Apply LoRA to a Linear layer.

    Args:
        linear: The linear layer to adapt
        config: LoRA configuration

    Returns:
        LoRA-wrapped linear layer
    """
    return LoRALinear(
        linear=linear,
        rank=config.rank,
        alpha=config.alpha,
    )