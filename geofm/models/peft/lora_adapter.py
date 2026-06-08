"""geofm.models.peft.lora_adapter

LoRA (Low-Rank Adaptation) layer implementation.
Replaces linear layers with low-rank decomposed adapters.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from typing import Optional


class LoRALinear(nn.Module):
    """LoRA-adapted linear layer.

    Replaces a base linear layer with:
    - Frozen original weights
    - Trainable low-rank decomposition (A @ B)

    The adaptation is: y = Wx + (alpha/rank) * (B @ A) @ x

    Attributes:
        base_layer: Original linear layer (frozen)
        rank: LoRA rank (lower = fewer params)
        alpha: Scaling factor
        lora_A: Low-rank matrix A (rank x in_features)
        lora_B: Low-rank matrix B (out_features x rank)
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
    ):
        """Initialize LoRA layer.

        Args:
            base_layer: Original linear layer to adapt
            rank: LoRA rank (default: 16)
            alpha: LoRA alpha scaling factor (default: 32)
            dropout: Dropout probability for LoRA inputs (default: 0.05)
        """
        super().__init__()

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # LoRA matrices
        # A: initialized with kaiming uniform (like linear layer)
        # B: initialized with zeros (so lora starts as identity)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Initialize A with kaiming
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Initialize B with zeros
        nn.init.zeros_(self.lora_B)

        # Freeze base layer
        for p in self.base_layer.parameters():
            p.requires_grad = False

        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation.

        Args:
            x: Input tensor (..., in_features)

        Returns:
            Output tensor (..., out_features)
        """
        # Base forward pass
        base = self.base_layer(x)

        # LoRA forward pass
        # x -> (..., in_features)
        # lora_A: (rank, in_features)
        # lora_B: (out_features, rank)
        # x @ lora_A.T: (..., rank)
        # ... @ lora_B.T: (..., out_features)
        x_drop = self.dropout(x)
        lora = torch.matmul(
            torch.matmul(x_drop, self.lora_A.t()), self.lora_B.t()
        )

        return base + self.scaling * lora

    def merge(self) -> nn.Linear:
        """Merge LoRA weights into base layer.

        Creates a new linear layer with merged weights.
        Useful for exporting or inference.

        Returns:
            Merged linear layer
        """
        # W_merged = W + (alpha/rank) * B @ A
        merged_weight = self.base_layer.weight + self.scaling * (self.lora_B @ self.lora_A)

        merged = nn.Linear(
            self.base_layer.in_features,
            self.base_layer.out_features,
            bias=self.base_layer.bias is not None,
        )
        merged.weight = nn.Parameter(merged_weight)

        if self.base_layer.bias is not None:
            merged.bias = self.base_layer.bias

        return merged

    @staticmethod
    def from_linear(
        linear: nn.Linear,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
    ) -> "LoRALinear":
        """Create LoRA layer from standard linear layer.

        Args:
            linear: Original linear layer
            rank: LoRA rank
            alpha: LoRA alpha
            dropout: Dropout probability

        Returns:
            LoRA-adapted layer
        """
        return LoRALinear(
            base_layer=linear,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )


class LoRAConv2d(nn.Module):
    """LoRA-adapted 2D convolution layer.

    Similar to LoRALinear but for convolutions.
    """

    def __init__(
        self,
        base_layer: nn.Conv2d,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
    ):
        """Initialize LoRA Conv2d layer.

        Args:
            base_layer: Original conv2d layer to adapt
            rank: LoRA rank
            alpha: LoRA alpha scaling factor
            dropout: Dropout probability
        """
        super().__init__()

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)

        in_channels = base_layer.in_channels
        out_channels = base_layer.out_channels
        kernel_size = base_layer.kernel_size

        # Flatten spatial dimensions for low-rank adaptation
        self.spatial_dims = kernel_size[0] * kernel_size[1]

        # LoRA matrices
        self.lora_A = nn.Parameter(
            torch.zeros(rank, in_channels * self.spatial_dims)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_channels * self.spatial_dims, rank)
        )

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Freeze base layer
        for p in self.base_layer.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Output tensor (B, C', H', W')
        """
        # Base forward
        base = self.base_layer(x)

        # LoRA adaptation
        # Unfold input
        x_unfold = nn.functional.unfold(
            x,
            self.base_layer.kernel_size,
            stride=self.base_layer.stride,
            padding=self.base_layer.padding,
            dilation=self.base_layer.dilation,
        )
        # x_unfold: (B, C*k*k, H'*W')

        # Apply LoRA
        # x_unfold @ lora_A.T @ lora_B.T
        lora = torch.matmul(
            torch.matmul(x_unfold, self.lora_A.t()), self.lora_B.t()
        )
        # lora: (B, out_channels*k*k, H'*W')

        # Reshape
        lora = lora.view(
            x.shape[0],
            self.base_layer.out_channels,
            self.spatial_dims,
            -1,
        )
        lora = lora.sum(dim=2)  # Sum over spatial dims

        return base + self.scaling * lora


def count_lora_parameters(model: nn.Module) -> int:
    """Count trainable LoRA parameters in model.

    Args:
        model: Model with LoRA layers

    Returns:
        Number of trainable LoRA parameters
    """
    total = 0
    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, LoRAConv2d)):
            total += module.lora_A.numel() + module.lora_B.numel()
    return total


def apply_lora_to_linear(
    model: nn.Module,
    target_names: list[str],
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
) -> None:
    """Apply LoRA to all linear layers matching target names.

    Args:
        model: Model to modify
        target_names: List of module name patterns to match
        rank: LoRA rank
        alpha: LoRA alpha
        dropout: Dropout probability
    """
    for name, module in model.named_modules():
        # Check if name matches any target
        if not any(target in name for target in target_names):
            continue

        # Check if module is linear
        if isinstance(module, nn.Linear):
            # Create LoRA version
            lora_layer = LoRALinear(
                base_layer=module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )

            # Replace in parent
            parent_name, child_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
            setattr(parent, child_name, lora_layer)


# Re-export TerraMindLoRA for backward compatibility
# This was moved from the original lora_adapter.py
try:
    from geofm.models.peft.lora_layer import TerraMindLoRA
    __all__ = ['LoRALinear', 'LoRAConv2d', 'count_lora_parameters', 'apply_lora_to_linear', 'TerraMindLoRA']
except ImportError:
    __all__ = ['LoRALinear', 'LoRAConv2d', 'count_lora_parameters', 'apply_lora_to_linear']