from __future__ import annotations

import torch.nn as nn


class AdapterBank(
    nn.Module
):
    """
    Holds one adapter
    per task.
    """

    def __init__(self):
        super().__init__()

        self.adapters = nn.ModuleDict()

    def register_task(
        self,
        task_name: str,
        adapter: nn.Module,
    ):

        self.adapters[
            task_name
        ] = adapter

    def get_adapter(
        self,
        task_name: str,
    ):

        return self.adapters[
            task_name
        ]

    def forward(
        self,
        task_name,
        features,
    ):

        adapter = self.get_adapter(
            task_name
        )

        return adapter(features)