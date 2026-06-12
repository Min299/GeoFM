"""geofm.data

Data utilities for GeoFM.

Provides datasets, dataloaders, and data processing utilities.

Includes:
- base_dataset: Abstract base class for all datasets
- dataset_factory: Factory for creating datasets
- task_dataset_router: Router for task-specific datasets
- multitask_dataset: Multi-task dataset wrapper
- dataset_statistics: Statistics computation
- sample_visualizer: Sample inspection utilities
- collate: Collate functions for batching
- samplers: Task sampling strategies
- dataloader_factory: Factory for creating dataloaders
- multitask_dataloader: Multi-task dataloader
"""
from geofm.data.base_dataset import (
    BaseGeoDataset,
    DatasetSplit,
)

from geofm.data.dataset_factory import (
    DatasetFactory,
    register_dataset,
)

from geofm.data.task_dataset_router import (
    TaskDatasetRouter,
    LazyTaskDatasetRouter,
)

from geofm.data.multitask_dataset import (
    MultiTaskDataset,
    ConcatMultiTaskDataset,
    CycleMultiTaskDataset,
)

from geofm.data.dataset_statistics import (
    DatasetStatistics,
    DatasetStats,
)

from geofm.data.sample_visualizer import (
    SampleVisualizer,
)

from geofm.data.collate import (
    geofm_collate,
    segmentation_collate,
    multitask_collate,
    variable_size_collate,
    ignore_none_collate,
)

from geofm.data.samplers import (
    RoundRobinSampler,
    RandomTaskSampler,
    WeightedTaskSampler,
    PriorityTaskSampler,
    BalancedTaskSampler,
)

from geofm.data.dataloader_factory import (
    DataLoaderFactory,
    create_dataloader,
)

from geofm.data.multitask_dataloader import (
    MultiTaskDataLoader,
    RoundRobinMultiTaskLoader,
    RandomMultiTaskLoader,
)


__all__ = [
    # Base
    "BaseGeoDataset",
    "DatasetSplit",
    # Factory
    "DatasetFactory",
    "register_dataset",
    # Router
    "TaskDatasetRouter",
    "LazyTaskDatasetRouter",
    # Multi-task Dataset
    "MultiTaskDataset",
    "ConcatMultiTaskDataset",
    "CycleMultiTaskDataset",
    # Statistics
    "DatasetStatistics",
    "DatasetStats",
    # Visualizer
    "SampleVisualizer",
    # Collate
    "geofm_collate",
    "segmentation_collate",
    "multitask_collate",
    "variable_size_collate",
    "ignore_none_collate",
    # Samplers
    "RoundRobinSampler",
    "RandomTaskSampler",
    "WeightedTaskSampler",
    "PriorityTaskSampler",
    "BalancedTaskSampler",
    # DataLoader
    "DataLoaderFactory",
    "create_dataloader",
    # Multi-task DataLoader
    "MultiTaskDataLoader",
    "RoundRobinMultiTaskLoader",
    "RandomMultiTaskLoader",
]