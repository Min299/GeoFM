from __future__ import annotations

TASK_REGISTRY = {
    "flood": {
        "task_type": "segmentation",
        "num_classes": 2,
    },
    "burn": {
        "task_type": "segmentation",
        "num_classes": 2,
    },
    "lulc": {
        "task_type": "segmentation",
        "num_classes": 10,
    },
    "crop": {
        "task_type": "classification",
        "num_classes": 13,
    },
    "ndvi": {
        "task_type": "regression",
        "num_outputs": 1,
    },
}


def get_task_config(task_name: str):

    if task_name not in TASK_REGISTRY:
        raise KeyError(
            f"Unknown task: {task_name}"
        )

    return TASK_REGISTRY[task_name]