"""geofm.integration.runtime

GeoFM training runtime.
"""
from __future__ import annotations

from geofm.integration.model_factory import (
    ModelFactory,
)

from geofm.training.fit import (
    fit,
)

from geofm.training.epoch_runner import (
    EpochRunner,
)


class Runtime:
    """Training runtime for GeoFM."""

    def __init__(
        self,
        cfg,
    ):
        """Initialize runtime with config.

        Args:
            cfg: Configuration object
        """
        self.cfg = cfg

    def run(
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
        model = (
            ModelFactory.build(
                self.cfg
            )
        )

        runner = (
            EpochRunner(
                model,
                optimizer,
                criterion,
            )
        )

        return fit(
            runner,
            train_loader,
            val_loader,
            epochs=self.cfg.training.epochs,
        )