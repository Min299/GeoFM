"""geofm.datasets

Geospatial dataset loaders with uniform sample contract.
Every dataset returns: {"image", "metadata", "task", "label"}
"""
from geofm.datasets.base_dataset import BaseGeoDataset
from geofm.datasets.registry import DATASET_REGISTRY, register_dataset, list_registered_tasks, is_registered
from geofm.datasets.builder import build_dataset, build_dataloader, DatasetBuildError
from geofm.datasets.transforms import build_train_transform, build_val_transform

__all__ = [
    "BaseGeoDataset",
    "DATASET_REGISTRY",
    "register_dataset",
    "list_registered_tasks",
    "is_registered",
    "build_dataset",
    "build_dataloader",
    "DatasetBuildError",
    "build_train_transform",
    "build_val_transform",
]
