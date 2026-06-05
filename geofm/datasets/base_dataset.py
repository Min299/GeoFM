"""geofm.datasets.base_dataset

Abstract base class for all geospatial datasets. Every concrete dataset
returns a uniform sample dict: modalities, metadata, task, label.

The modalities dict follows TerraTorch's modality format:
    {
        "S2L1C": tensor,   # Sentinel-2 Level-1C
        "S1GRD": tensor,   # Sentinel-1 SAR
        "S2L2A": tensor,   # Sentinel-2 Level-2A
        ...
    }
"""
from abc import ABC, abstractmethod
from pathlib import Path

from torch.utils.data import Dataset


class BaseGeoDataset(Dataset, ABC):
    def __init__(self, root_dir, task, modalities=None):
        self.root_dir = Path(root_dir)
        self.task = task
        # Default modalities if not specified
        self.modalities = modalities or ["S2L1C"]

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def load_modalities(self, idx):
        """Load all modality tensors for a sample.

        Returns:
            dict: {"modality_name": tensor, ...}
                  e.g., {"S2L1C": tensor, "S1GRD": tensor}
        """
        pass

    @abstractmethod
    def load_label(self, idx):
        pass

    @abstractmethod
    def load_metadata(self, idx):
        pass

    def __getitem__(self, idx):
        modalities = self.load_modalities(idx)
        label = self.load_label(idx)
        metadata = self.load_metadata(idx)

        return {
            "modalities": modalities,
            "metadata": metadata,
            "task": self.task,
            "label": label,
        }
