"""geofm.models.tasks.classification_model

Classification model using TerraMind backbone.
"""
from __future__ import annotations

import torch.nn as nn


class ClassificationModel(
    nn.Module
):
    """Classification model with TerraMind backbone.

    Architecture:
        Input Batch
            ->
        TerraMind Backbone
            ->
        Global Average Pool
            ->
        Classifier
            ->
        Classification Output
    """

    def __init__(
        self,
        backbone,
        classifier,
    ):
        super().__init__()

        self.backbone = (
            backbone
        )

        self.classifier = (
            classifier
        )

    def forward(
        self,
        batch,
    ):
        """Forward pass for classification.

        Args:
            batch: Input batch dictionary

        Returns:
            Classification logits
        """
        features = (
            self.backbone(
                batch
            )
        )

        x = features[-1]

        x = x.mean(
            dim=1
        )

        return self.classifier(
            x
        )