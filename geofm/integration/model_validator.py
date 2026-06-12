"""geofm.integration.model_validator

Validators for GeoFM model structure.
"""
from __future__ import annotations

from typing import Any


class ModelValidator:
    """Validates GeoFM model structure.

    Ensures models have required components:
    - backbone
    - adapter_bank
    - decoder_bank
    """

    REQUIRED_COMPONENTS = [
        "backbone",
        "adapter_bank",
        "decoder_bank",
    ]

    @staticmethod
    def validate_model(model: Any) -> None:
        """Validate model has required components.

        Args:
            model: GeoFM model instance

        Raises:
            ValueError: If model is missing required components
        """
        for attr in ModelValidator.REQUIRED_COMPONENTS:
            if not hasattr(model, attr):
                raise ValueError(
                    f"Model missing required component: {attr}"
                )

    @staticmethod
    def validate_task(model: Any, task: str) -> None:
        """Validate task is registered in model.

        Args:
            model: GeoFM model instance
            task: Task name

        Raises:
            ValueError: If task is not registered
        """
        if not hasattr(model.decoder_bank, "__contains__"):
            raise ValueError(
                "decoder_bank must support 'in' operator for task check"
            )

        if task not in model.decoder_bank:
            available = list(model.decoder_bank.keys()) if hasattr(model.decoder_bank, 'keys') else []
            raise ValueError(
                f"Task '{task}' not registered. Available: {available}"
            )

    @staticmethod
    def validate_forward_method(model: Any) -> None:
        """Validate model has valid forward method.

        Args:
            model: GeoFM model instance

        Raises:
            ValueError: If forward method is invalid
        """
        if not callable(getattr(model, 'forward', None)):
            raise ValueError("Model must have callable forward method")

    @staticmethod
    def validate_all(model: Any, task: str = None) -> None:
        """Run all validations.

        Args:
            model: GeoFM model instance
            task: Optional task name to validate

        Raises:
            ValueError: If any validation fails
        """
        ModelValidator.validate_model(model)
        ModelValidator.validate_forward_method(model)

        if task is not None:
            ModelValidator.validate_task(model, task)