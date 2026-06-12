"""geofm.builders.trainer_builder

Builder for creating trainers.
"""
from __future__ import annotations

import torch


class TrainerBuilder:
    """Builder for creating trainers.

    Usage:
        trainer = TrainerBuilder.build("single", model=model, ...)
    """

    @staticmethod
    def build(
        mode: str,
        model,
        train_loader=None,
        val_loader=None,
        test_loader=None,
        device: str = "auto",
        **kwargs,
    ):
        """Build a trainer for the specified mode.

        Args:
            mode: Training mode (single, multitask, shared)
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            device: Device to train on
            **kwargs: Additional trainer arguments

        Returns:
            Trainer instance
        """
        from geofm.factories.trainer_factory import TrainerFactory

        return TrainerFactory.build(
            mode=mode,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            **kwargs,
        )

    @staticmethod
    def build_from_config(config, model):
        """Build trainer from experiment config.

        Args:
            config: ExperimentConfig object
            model: Model to train

        Returns:
            Trainer instance

        Raises:
            NotImplementedError: Config-based trainer building not yet implemented
        """
        raise NotImplementedError("Config-based trainer building pending.")