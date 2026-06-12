"""geofm.integration.pipeline

GeoFM end-to-end pipeline.
"""
from __future__ import annotations

from geofm.integration.runtime import (
    Runtime,
)


class GeoFMPipeline:
    """Single orchestration object for GeoFM.

    Wraps Runtime for config-driven training.

    Usage:
        pipeline = GeoFMPipeline(cfg)
        history = pipeline.train(train_loader, val_loader, optimizer, criterion)
    """

    def __init__(
        self,
        cfg,
    ):
        """Initialize pipeline with config.

        Args:
            cfg: Configuration object
        """
        self.runtime = Runtime(
            cfg
        )

    def train(
        self,
        train_loader,
        val_loader,
        optimizer,
        criterion,
    ):
        """Run training.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            criterion: Loss criterion

        Returns:
            Training history dict
        """
        return self.runtime.train(
            train_loader,
            val_loader,
            optimizer,
            criterion,
        )