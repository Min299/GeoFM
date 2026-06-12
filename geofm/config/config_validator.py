"""geofm.config.config_validator

Configuration validation utilities.

Validates that experiment configurations have required fields
and valid values.
"""
from __future__ import annotations

from typing import List, Dict, Any, Union
from geofm.config.experiment_config import ExperimentConfig


class ConfigValidator:
    """Validator for experiment configurations.

    Ensures that configurations have all required fields
    and valid values.

    Usage:
        config = ExperimentConfig(...)

        if ConfigValidator.validate(config):
            print("Config is valid!")
    """

    REQUIRED_FIELDS = [
        "experiment_name",
        "task",
        "model_type",
        "adapter_type",
    ]

    VALID_TASKS = [
        "flood",
        "burn",
        "lulc",
        "crop",
        "ndvi",
        "multitask",
    ]

    VALID_MODEL_TYPES = [
        "shared",
        "separate",
    ]

    VALID_ADAPTER_TYPES = [
        "feature",
        "lora",
        "hybrid",
        "fullft",
    ]

    @classmethod
    def validate(cls, config: Union[ExperimentConfig, Dict]) -> bool:
        """Validate a configuration.

        Args:
            config: ExperimentConfig or dict to validate

        Returns:
            True if valid

        Raises:
            ValueError: If configuration is invalid
        """
        # Convert to dict if needed
        if isinstance(config, ExperimentConfig):
            config_dict = config.to_dict()
        else:
            config_dict = config

        # Check required fields
        cls._validate_required_fields(config_dict)

        # Check task
        if "task" in config_dict:
            cls._validate_task(config_dict["task"])

        # Check model type
        if "model_type" in config_dict:
            cls._validate_model_type(config_dict["model_type"])

        # Check adapter type
        if "adapter_type" in config_dict:
            cls._validate_adapter_type(config_dict["adapter_type"])

        # Check numeric values
        cls._validate_numeric(config_dict)

        return True

    @classmethod
    def _validate_required_fields(cls, config: Dict) -> None:
        """Validate required fields are present.

        Args:
            config: Configuration dict

        Raises:
            ValueError: If required fields are missing
        """
        missing = []

        for field in cls.REQUIRED_FIELDS:
            if field not in config or config[field] is None:
                missing.append(field)

        if missing:
            raise ValueError(f"Missing required fields: {missing}")

    @classmethod
    def _validate_task(cls, task: str) -> None:
        """Validate task name.

        Args:
            task: Task name

        Raises:
            ValueError: If task is invalid
        """
        if task not in cls.VALID_TASKS:
            raise ValueError(
                f"Invalid task: '{task}'. Must be one of: {cls.VALID_TASKS}"
            )

    @classmethod
    def _validate_model_type(cls, model_type: str) -> None:
        """Validate model type.

        Args:
            model_type: Model type

        Raises:
            ValueError: If model type is invalid
        """
        if model_type not in cls.VALID_MODEL_TYPES:
            raise ValueError(
                f"Invalid model_type: '{model_type}'. "
                f"Must be one of: {cls.VALID_MODEL_TYPES}"
            )

    @classmethod
    def _validate_adapter_type(cls, adapter_type: str) -> None:
        """Validate adapter type.

        Args:
            adapter_type: Adapter type

        Raises:
            ValueError: If adapter type is invalid
        """
        if adapter_type not in cls.VALID_ADAPTER_TYPES:
            raise ValueError(
                f"Invalid adapter_type: '{adapter_type}'. "
                f"Must be one of: {cls.VALID_ADAPTER_TYPES}"
            )

    @classmethod
    def _validate_numeric(cls, config: Dict) -> None:
        """Validate numeric fields.

        Args:
            config: Configuration dict

        Raises:
            ValueError: If numeric values are invalid
        """
        # Check epochs
        if "epochs" in config:
            if not isinstance(config["epochs"], int) or config["epochs"] <= 0:
                raise ValueError("epochs must be a positive integer")

        # Check batch_size
        if "batch_size" in config:
            if not isinstance(config["batch_size"], int) or config["batch_size"] <= 0:
                raise ValueError("batch_size must be a positive integer")

        # Check learning_rate
        if "learning_rate" in config:
            if not isinstance(config["learning_rate"], (int, float)):
                raise ValueError("learning_rate must be a number")
            if config["learning_rate"] <= 0:
                raise ValueError("learning_rate must be positive")

    @classmethod
    def get_validation_report(cls, config: Union[ExperimentConfig, Dict]) -> Dict:
        """Get detailed validation report.

        Args:
            config: Configuration to validate

        Returns:
            Dictionary with validation results
        """
        report = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }

        # Convert to dict if needed
        if isinstance(config, ExperimentConfig):
            config_dict = config.to_dict()
        else:
            config_dict = config

        # Check required fields
        missing = []
        for field in cls.REQUIRED_FIELDS:
            if field not in config_dict or config_dict[field] is None:
                missing.append(field)

        if missing:
            report["valid"] = False
            report["errors"].append(f"Missing required fields: {missing}")

        # Check task
        if "task" in config_dict:
            if config_dict["task"] not in cls.VALID_TASKS:
                report["valid"] = False
                report["errors"].append(f"Invalid task: {config_dict['task']}")

        return report


def validate_config(config: ExperimentConfig) -> bool:
    """Convenience function to validate a config.

    Args:
        config: Configuration to validate

    Returns:
        True if valid

    Raises:
        ValueError: If invalid
    """
    return ConfigValidator.validate(config)