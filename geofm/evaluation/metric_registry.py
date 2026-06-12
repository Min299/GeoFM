"""geofm.evaluation.metric_registry

Registry of metrics for each task type.
"""


METRIC_REGISTRY = {
    "flood": ["iou", "dice", "f1", "precision", "recall"],
    "burn": ["iou", "dice", "f1", "precision", "recall"],
    "lulc": ["accuracy", "per_class_accuracy", "mean_class_accuracy"],
    "crop": ["accuracy"],
    "ndvi": ["rmse", "mae", "mse"],
}


def get_metrics(task: str) -> list[str]:
    """Get list of metrics for a task.

    Args:
        task: Task name

    Returns:
        List of metric names

    Raises:
        ValueError: If task not in registry
    """
    if task not in METRIC_REGISTRY:
        raise ValueError(f"Unknown task: {task}. Available: {list(METRIC_REGISTRY.keys())}")

    return METRIC_REGISTRY[task]


def get_primary_metric(task: str) -> str:
    """Get primary metric for a task.

    Args:
        task: Task name

    Returns:
        Primary metric name
    """
    if task in ["flood", "burn"]:
        return "iou"
    elif task in ["lulc", "crop"]:
        return "accuracy"
    elif task == "ndvi":
        return "rmse"
    else:
        return get_metrics(task)[0]


def is_segmentation_task(task: str) -> bool:
    """Check if task is a segmentation task.

    Args:
        task: Task name

    Returns:
        True if segmentation task
    """
    return task in ["flood", "burn"]


def is_classification_task(task: str) -> bool:
    """Check if task is a classification task.

    Args:
        task: Task name

    Returns:
        True if classification task
    """
    return task in ["lulc", "crop"]


def is_regression_task(task: str) -> bool:
    """Check if task is a regression task.

    Args:
        task: Task name

    Returns:
        True if regression task
    """
    return task == "ndvi"