"""geofm.builders.dataset_builder

Builder for creating datasets.
"""
from __future__ import annotations

from typing import Optional


class DatasetBuilder:
    """Builder for creating datasets.

    Usage:
        dataset = DatasetBuilder.build("flood", root="data/")
    """

    @staticmethod
    def build(
        dataset_name: str,
        root: Optional[str] = None,
        split: str = "train",
        transform=None,
        **kwargs,
    ):
        """Build a dataset.

        Args:
            dataset_name: Name of the dataset (flood, burn, lulc, etc.)
            root: Root directory of the dataset
            split: Dataset split (train, val, test)
            transform: Optional transforms to apply
            **kwargs: Additional dataset arguments

        Returns:
            Dataset instance

        Raises:
            NotImplementedError: Dataset integration pending
        """
        raise NotImplementedError(
            f"Dataset '{dataset_name}' integration pending. "
            "Please implement dataset loading for your specific dataset."
        )

    @staticmethod
    def list_available_datasets() -> list[str]:
        """List available dataset names.

        Returns:
            List of supported dataset names
        """
        return [
            "flood",
            "burn",
            "lulc",
            "crop",
            "ndvi",
        ]

    @staticmethod
    def validate_dataset_path(dataset_name: str, root: str) -> bool:
        """Validate that dataset path exists.

        Args:
            dataset_name: Name of the dataset
            root: Root directory

        Returns:
            True if valid, False otherwise
        """
        from pathlib import Path

        dataset_root = Path(root) / dataset_name

        if not dataset_root.exists():
            return False

        # Check for expected subdirectories
        expected_splits = ["train", "val", "test"]
        has_split = any(
            (dataset_root / split).exists()
            for split in expected_splits
        )

        return has_split