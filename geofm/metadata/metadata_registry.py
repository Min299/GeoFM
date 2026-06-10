"""geofm.metadata.metadata_registry

Registry for dataset metadata field mappings.
"""
from __future__ import annotations


class MetadataRegistry:
    """Registry for mapping dataset names to their metadata fields."""

    def __init__(self):
        """Initialize empty registry."""
        self._registry = {}

    def register(
        self,
        dataset_name: str,
        fields: list[str],
    ):
        """Register metadata fields for a dataset.

        Args:
            dataset_name: Name of the dataset
            fields: List of metadata field names
        """
        self._registry[
            dataset_name
        ] = fields

    def get_fields(
        self,
        dataset_name: str,
    ):
        """Get metadata fields for a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            List of field names, empty list if not registered
        """
        return self._registry.get(
            dataset_name,
            [],
        )

    def datasets(self):
        """Get list of registered dataset names.

        Returns:
            List of dataset names
        """
        return list(
            self._registry.keys()
        )