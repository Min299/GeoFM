"""geofm.training

Training utilities for multi-task learning.

Includes:
- multitask_step: Core training step
- multitask_trainer: Multi-task training loop
- epoch_runner: Epoch-level training and validation
- fit: High-level training loop with history tracking
- validate: Validation utilities
- predict: Prediction utilities
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

from geofm.training.epoch_runner import (
    EpochRunner,
)

from geofm.training.fit import (
    fit,
)

from geofm.training.validate import (
    validate,
)

from geofm.training.predict import (
    predict,
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
    # Epoch-level
    "EpochRunner",
    "fit",
    "validate",
    "predict",
]