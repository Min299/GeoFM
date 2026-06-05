"""geofm.datasets.base_dataset

Abstract base class for all geospatial datasets. Every concrete dataset
returns a uniform sample dict: image, metadata, task, label.
"""
from abc import ABC, abstractmethod

from torch.utils.data import Dataset


class BaseGeoDataset(Dataset, ABC):
    def __init__(self, root_dir, task):
        self.root_dir = root_dir
        self.task = task

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def load_image(self, idx):
        pass

    @abstractmethod
    def load_label(self, idx):
        pass

    @abstractmethod
    def load_metadata(self, idx):
        pass

    def __getitem__(self, idx):
        image = self.load_image(idx)
        label = self.load_label(idx)
        metadata = self.load_metadata(idx)

        return {
            "image": image,
            "metadata": metadata,
            "task": self.task,
            "label": label,
        }
