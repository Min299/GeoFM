"""geofm.data.dataset_factory

Factory for creating and managing datasets.

Provides registration and building interface for datasets.
"""
from __future__ import annotations

from typing import Dict, Type, Any, Optional, List


class DatasetFactory:
    """Factory for creating datasets by name.

    Provides a registry pattern for dataset classes.

    Usage:
        # Register a dataset
        DatasetFactory.register("flood", FloodDataset)

        # Build a dataset
        dataset = DatasetFactory.build("flood", split="train")

        # List available datasets
        available = DatasetFactory.available()
    """

    _registry: Dict[str, Type] = {}

    @classmethod
    def register(
        cls,
        name: str,
        dataset_cls: Type,
    ) -> None:
        """Register a dataset class.

        Args:
            name: Dataset name (e.g., "flood", "burn")
            dataset_cls: Dataset class to register
        """
        cls._registry[name] = dataset_cls

    @classmethod
    def build(
        cls,
        name: str,
        **kwargs,
    ) -> Any:
        """Build a dataset by name.

        Args:
            name: Dataset name
            **kwargs: Arguments to pass to dataset constructor

        Returns:
            Instance of the dataset class

        Raises:
            ValueError: If dataset name is not registered
        """
        if name not in cls._registry:
            available = cls.available()
            raise ValueError(
                f"Unknown dataset: '{name}'. Available: {available}"
            )

        return cls._registry[name](**kwargs)

    @classmethod
    def available(cls) -> List[str]:
        """Get list of available dataset names.

        Returns:
            List of registered dataset names
        """
        return list(cls._registry.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a dataset is registered.

        Args:
            name: Dataset name

        Returns:
            True if registered
        """
        return name in cls._registry

    @classmethod
    def unregister(cls, name: str) -> bool:
        """Unregister a dataset.

        Args:
            name: Dataset name

        Returns:
            True if was registered and now removed
        """
        if name in cls._registry:
            del cls._registry[name]
            return True
        return False

    @classmethod
    def clear(cls) -> None:
        """Clear all registered datasets."""
        cls._registry.clear()

    @classmethod
    def register_multiple(cls, datasets: Dict[str, Type]) -> None:
        """Register multiple datasets at once.

        Args:
            datasets: Dictionary mapping name to dataset class
        """
        for name, dataset_cls in datasets.items():
            cls.register(name, dataset_cls)


def register_dataset(name: str):
    """Decorator to register a dataset class.

    Usage:
        @register_dataset("flood")
        class FloodDataset(BaseGeoDataset):
            ...
    """
    def decorator(cls):
        DatasetFactory.register(name, cls)
        return cls
    return decorator