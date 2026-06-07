from __future__ import annotations

import torch
import torch.nn as nn


class SharedGeoFM(nn.Module):
    """
    Shared Encoder

        ↓

    Adapter Bank

        ↓

    Decoder Bank

        ↓

    Task Prediction
    """

    def __init__(
        self,
        backbone: nn.Module,
        adapter_bank: nn.Module,
        decoder_bank: nn.Module,
    ):
        super().__init__()

        self.backbone = backbone

        self.adapter_bank = adapter_bank

        self.decoder_bank = decoder_bank

    def extract_features(
        self,
        batch,
    ):
        """
        Backbone output:
            [F2,F5,F8,F11]
        """

        if hasattr(
            self.backbone,
            "extract_features",
        ):
            return self.backbone.extract_features(
                batch
            )

        return self.backbone(batch)

    def forward(
        self,
        batch,
        task_name: str,
    ):

        features = self.extract_features(
            batch
        )

        features = self.adapter_bank(
            task_name,
            features,
        )

        prediction = self.decoder_bank(
            task_name,
            features,
        )

        return prediction