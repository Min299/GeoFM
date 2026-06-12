"""geofm.data.base_dataset

Base dataset class for all GeoFM datasets.

Provides a common contract that all task-specific datasets inherit from.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from torch.utils.data import Dataset


class BaseGeoDataset(Dataset, ABC):
    """Abstract base class for all GeoFM datasets.

    All task-specific datasets (Flood, Burn, LULC, Crop, NDVI) should
    inherit from this class to ensure a consistent interface.

    Usage:
        class FloodDataset(BaseGeoDataset):
            def __len__(self):
                return 100

            def __getitem__(self, idx):
                return {"image": ..., "mask": ...}
    """

    def __init__(
        self,
        split: str,
        root_dir: Optional[str] = None,
    ):
        """Initialize base dataset.

        Args:
            split: Dataset split ("train", "val", "test")
            root_dir: Optional root directory path
        """
        self.split = split
        self.root_dir = root_dir

    @abstractmethod
    def __len__(self) -> int:
        """Get dataset length.

        Returns:
            Number of samples in dataset
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get sample by index.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing sample data
        """
        raise NotImplementedError

    @property
    def task_name(self) -> str:
        """Get task name from class name.

        Returns:
            Task name (e.g., "FloodDataset" -> "flood")
        """
        name = self.__class__.__name__
        # Remove "Dataset" suffix if present
        if name.endswith("Dataset"):
            name = name[:-7]
        return name.lower()

    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get metadata about a sample without loading full data.

        Args:
            idx: Sample index

        Returns:
            Dictionary with sample metadata
        """
        return {
            "index": idx,
            "task": self.task_name,
            "split": self.split,
        }

    def validate_index(self, idx: int) -> bool:
        """Check if index is valid.

        Args:
            idx: Index to validate

        Returns:
            True if valid, False otherwise
        """
        return 0 <= idx < len(self)


class DatasetSplit:
    """Dataset split enumeration."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @classmethod
    def all(cls) -> list:
        """Get all split values.

        Returns:
            List of all splits
        """
        return [cls.TRAIN, cls.VAL, cls.TEST]

    @classmethod
    def is_valid(cls, split: str) -> bool:
        """Check if split is valid.

        Args:
            split: Split string to check

        Returns:
            True if valid split
        """
        return split in cls.all()