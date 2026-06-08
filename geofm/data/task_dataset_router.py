"""geofm.data.task_dataset_router

Router for accessing task-specific datasets.

Provides a unified interface for accessing multiple task datasets.
"""
from __future__ import annotations

from typing import Dict, Any, List, Optional


class TaskDatasetRouter:
    """Router for accessing task-specific datasets.

    Provides a unified interface for managing multiple task datasets.

    Usage:
        router = TaskDatasetRouter({
            "flood": flood_dataset,
            "burn": burn_dataset,
        })

        dataset = router.get_dataset("flood")
        tasks = router.available_tasks()
    """

    def __init__(
        self,
        datasets: Dict[str, Any],
    ):
        """Initialize router with datasets.

        Args:
            datasets: Dictionary mapping task name to dataset
        """
        self.datasets = datasets

    def get_dataset(self, task: str) -> Any:
        """Get dataset for a specific task.

        Args:
            task: Task name

        Returns:
            Dataset for the task

        Raises:
            KeyError: If task is not registered
        """
        if task not in self.datasets:
            available = self.available_tasks()
            raise KeyError(
                f"Task '{task}' not found. Available: {available}"
            )

        return self.datasets[task]

    def available_tasks(self) -> List[str]:
        """Get list of available task names.

        Returns:
            List of task names
        """
        return list(self.datasets.keys())

    def has_task(self, task: str) -> bool:
        """Check if a task is registered.

        Args:
            task: Task name

        Returns:
            True if task exists
        """
        return task in self.datasets

    def add_dataset(self, task: str, dataset: Any) -> None:
        """Add a dataset for a task.

        Args:
            task: Task name
            dataset: Dataset instance
        """
        self.datasets[task] = dataset

    def remove_dataset(self, task: str) -> bool:
        """Remove a dataset for a task.

        Args:
            task: Task name

        Returns:
            True if was present and now removed
        """
        if task in self.datasets:
            del self.datasets[task]
            return True
        return False

    def task_count(self) -> int:
        """Get number of registered tasks.

        Returns:
            Number of tasks
        """
        return len(self.datasets)

    def get_all_lengths(self) -> Dict[str, int]:
        """Get lengths of all datasets.

        Returns:
            Dictionary mapping task to dataset length
        """
        return {
            task: len(dataset)
            for task, dataset in self.datasets.items()
        }

    def get_total_samples(self) -> int:
        """Get total number of samples across all tasks.

        Returns:
            Total sample count
        """
        return sum(len(ds) for ds in self.datasets.values())

    def get_tasks_with_min_samples(self, min_samples: int) -> List[str]:
        """Get tasks that have at least min_samples.

        Args:
            min_samples: Minimum number of samples

        Returns:
            List of task names meeting the threshold
        """
        return [
            task
            for task, dataset in self.datasets.items()
            if len(dataset) >= min_samples
        ]


class LazyTaskDatasetRouter(TaskDatasetRouter):
    """Router that loads datasets lazily.

    Useful for large datasets where loading all at once is expensive.
    """

    def __init__(
        self,
        dataset_factories: Dict[str, Any],
    ):
        """Initialize lazy router.

        Args:
            dataset_factories: Dictionary mapping task name to factory function
        """
        self.dataset_factories = dataset_factories
        self.datasets: Dict[str, Any] = {}
        self._loaded = set()

    def get_dataset(self, task: str) -> Any:
        """Get dataset, loading it if necessary.

        Args:
            task: Task name

        Returns:
            Dataset for the task
        """
        if task not in self._loaded:
            if task not in self.dataset_factories:
                raise KeyError(f"Task '{task}' not in factories")
            self.datasets[task] = self.dataset_factories[task]()
            self._loaded.add(task)

        return self.datasets[task]

    def preload_all(self) -> None:
        """Preload all datasets."""
        for task in self.dataset_factories:
            self.get_dataset(task)

    def is_loaded(self, task: str) -> bool:
        """Check if a dataset is loaded.

        Args:
            task: Task name

        Returns:
            True if loaded
        """
        return task in self._loaded

    def unload(self, task: str) -> None:
        """Unload a dataset to free memory.

        Args:
            task: Task name
        """
        if task in self.datasets:
            del self.datasets[task]
        if task in self._loaded:
            self._loaded.remove(task)