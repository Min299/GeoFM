"""geofm.factories.trainer_factory

Factory for creating trainer modules.
"""
from __future__ import annotations


class TrainerFactory:
    """Factory for creating trainer modules.

    Usage:
        trainer = TrainerFactory.build("single", model=model, ...)
    """

    @staticmethod
    def build(
        mode: str,
        **kwargs,
    ):
        """Build a trainer for the specified mode.

        Args:
            mode: Training mode (single, multitask)
            **kwargs: Trainer-specific arguments

        Returns:
            Trainer instance

        Raises:
            ValueError: If mode is unknown
            NotImplementedError: If trainer is not yet implemented
        """
        if mode == "single":
            # Use FineTuneTrainer for single-task training
            from geofm.trainers.finetune_trainer import FineTuneTrainer
            return FineTuneTrainer(**kwargs)

        if mode == "multitask":
            from geofm.trainers.multitask_trainer import MultiTaskTrainer
            return MultiTaskTrainer(**kwargs)

        if mode == "shared":
            from geofm.trainers.shared_trainer import SharedTrainer
            return SharedTrainer(**kwargs)

        raise ValueError(f"Unknown training mode: {mode}")