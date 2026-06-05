"""geofm.datasets.registry

Central registry mapping task names to dataset implementations.
All datasets must follow the BaseGeoDataset contract:
    dataset[idx] -> {"image", "metadata", "task", "label"}
"""
from geofm.datasets.flood.sen1floods11 import Sen1Floods11Dataset

DATASET_REGISTRY = {
    "flood": Sen1Floods11Dataset,
}


def register_dataset(task_name, dataset_cls=None):
    """Decorator or function to register a dataset class for a task.

    Usage as decorator:
        @register_dataset("flood")
        class MyDataset(BaseGeoDataset):
            ...

    Usage as function:
        register_dataset("flood", MyDataset)
    """
    def decorator(cls):
        DATASET_REGISTRY[task_name] = cls
        return cls

    if dataset_cls is not None:
        # Called as function: register_dataset("flood", MyDataset)
        DATASET_REGISTRY[task_name] = dataset_cls
        return dataset_cls

    # Called as decorator: @register_dataset("flood")
    return decorator


def list_registered_tasks():
    """Return list of all registered task names."""
    return list(DATASET_REGISTRY.keys())


def is_registered(task_name):
    """Check if a task is registered."""
    return task_name in DATASET_REGISTRY