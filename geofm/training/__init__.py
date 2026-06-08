"""geofm.training

Training utilities for multi-task learning.

Includes:
- multitask_step: Core training step
- multitask_trainer: Multi-task training loop
"""
from geofm.training.multitask_step import (
    multitask_step,
    multitask_step_with_grad,
    multitask_step_mixed_precision,
    MultiTaskStepper,
)

from geofm.training.multitask_trainer import (
    MultiTaskTrainer,
    GradientMultiTaskTrainer,
    AdaptiveMultiTaskTrainer,
    TrainingMetrics,
    EpochMetrics,
)


__all__ = [
    # Step functions
    "multitask_step",
    "multitask_step_with_grad",
    "multitask_step_mixed_precision",
    "MultiTaskStepper",
    # Trainers
    "MultiTaskTrainer",
    "GradientMultiTaskTrainer",
    "AdaptiveMultiTaskTrainer",
    # Metrics
    "TrainingMetrics",
    "EpochMetrics",
]