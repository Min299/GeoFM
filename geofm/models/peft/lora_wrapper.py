"""geofm.models.peft.lora_wrapper

LoRA wrapper for fine-tuning.
"""
from __future__ import annotations

import torch.nn as nn


class LoRAWrapper(
    nn.Module
):
    """Wrapper for LoRA fine-tuning.

    Wraps a model with LoRA adaptation for parameter-efficient
    fine-tuning while keeping the original model frozen.
    """

    def __init__(
        self,
        model,
    ):
        super().__init__()

        self.model = model

    def forward(
        self,
        *args,
        **kwargs,
    ):
        return self.model(
            *args,
            **kwargs,
        )