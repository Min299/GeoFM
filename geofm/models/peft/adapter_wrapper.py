"""geofm.models.peft.adapter_wrapper

Adapter wrapper for fine-tuning.
"""
from __future__ import annotations

import torch.nn as nn


class AdapterWrapper(
    nn.Module
):
    """Wrapper for adapter-based fine-tuning.

    Wraps a model with task-specific adapters for parameter-efficient
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