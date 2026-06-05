"""geofm.datasets.builder

Universal dataset factory for creating dataset instances by task name.
"""
from geofm.datasets.registry import DATASET_REGISTRY, is_registered


class DatasetBuildError(Exception):
    """Raised when dataset building fails."""
    pass


def build_dataset(task, root_dir, metadata_csv=None, transform=None, **kwargs):
    """Build a dataset instance for the given task.

    Args:
        task: Task name (e.g., "flood", "crop", "burn")
        root_dir: Root directory containing dataset files
        metadata_csv: Optional path to metadata CSV file
        transform: Optional transform to apply to images
        **kwargs: Additional arguments passed to dataset constructor

    Returns:
        Dataset instance

    Raises:
        DatasetBuildError: If task is not registered or build fails
    """
    if not is_registered(task):
        available = list(DATASET_REGISTRY.keys())
        raise DatasetBuildError(
            f"Task '{task}' not registered. Available: {available}"
        )

    dataset_cls = DATASET_REGISTRY[task]

    try:
        return dataset_cls(
            root_dir=root_dir,
            metadata_csv=metadata_csv,
            transform=transform,
            **kwargs
        )
    except Exception as e:
        raise DatasetBuildError(
            f"Failed to build dataset for task '{task}': {e}"
        ) from e


def build_dataloader(task, root_dir, batch_size=16, shuffle=False,
                     num_workers=4, metadata_csv=None, transform=None, **kwargs):
    """Build a DataLoader for the given task.

    Args:
        task: Task name
        root_dir: Root directory containing dataset files
        batch_size: Batch size (default: 16)
        shuffle: Whether to shuffle data (default: False)
        num_workers: Number of worker processes (default: 4)
        metadata_csv: Optional path to metadata CSV
        transform: Optional transform
        **kwargs: Additional arguments passed to DataLoader

    Returns:
        torch.utils.data.DataLoader instance
    """
    import torch.utils.data as data

    dataset = build_dataset(
        task=task,
        root_dir=root_dir,
        metadata_csv=metadata_csv,
        transform=transform
    )

    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs
    )