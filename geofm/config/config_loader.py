"""geofm.config.config_loader

Configuration file loader.

Handles loading experiment configs from YAML files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Union
import yaml

from geofm.config.experiment_config import (
    ExperimentConfig,
    MultiTaskExperimentConfig,
)


class ConfigLoader:
    """Loader for configuration files.

    Handles loading YAML configuration files and converting them
    to ExperimentConfig objects.

    Usage:
        config = ConfigLoader.load("configs/flood.yaml")

        # Or load from dict
        config = ConfigLoader.load_dict({"task": "flood", ...})
    """

    @staticmethod
    def load(path: Union[str, Path]) -> ExperimentConfig:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML config file

        Returns:
            ExperimentConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML is invalid
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return ConfigLoader.load_dict(data)

    @staticmethod
    def load_dict(data: Dict[str, Any]) -> ExperimentConfig:
        """Load configuration from dictionary.

        Args:
            data: Dictionary with config data

        Returns:
            ExperimentConfig instance
        """
        # Check if multi-task config
        if "tasks" in data:
            return MultiTaskExperimentConfig(**data)
        else:
            return ExperimentConfig(**data)

    @staticmethod
    def load_multiple(paths: list) -> Dict[str, ExperimentConfig]:
        """Load multiple configuration files.

        Args:
            paths: List of paths to config files

        Returns:
            Dictionary mapping filename to config
        """
        configs = {}

        for path in paths:
            path = Path(path)
            name = path.stem
            configs[name] = ConfigLoader.load(path)

        return configs

    @staticmethod
    def save(config: ExperimentConfig, path: Union[str, Path]) -> None:
        """Save configuration to YAML file.

        Args:
            config: ExperimentConfig to save
            path: Output path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False)


class ConfigMerger:
    """Utility for merging configurations."""

    @staticmethod
    def merge(base: Dict, override: Dict) -> Dict:
        """Merge two config dictionaries.

        Args:
            base: Base configuration
            override: Override configuration

        Returns:
            Merged configuration
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigMerger.merge(result[key], value)
            else:
                result[key] = value

        return result

    @staticmethod
    def merge_configs(
        base: ExperimentConfig,
        override: Dict[str, Any],
    ) -> ExperimentConfig:
        """Merge config with override dict.

        Args:
            base: Base configuration
            override: Override dictionary

        Returns:
            Merged configuration
        """
        base_dict = base.to_dict()
        merged = ConfigMerger.merge(base_dict, override)
        return ConfigLoader.load_dict(merged)