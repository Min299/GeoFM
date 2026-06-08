"""geofm.data.multitask_dataloader

Multi-task data loader that provides access to multiple task loaders.

Provides a unified interface for loading data from multiple tasks.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Any, Iterator
from torch.utils.data import DataLoader


class MultiTaskDataLoader:
    """DataLoader that manages multiple task-specific DataLoaders.

    Provides a unified interface for multi-task training.

    Usage:
        loaders = {
            "flood": flood_loader,
            "burn": burn_loader,
        }

        mt_loader = MultiTaskDataLoader(loaders)

        loader = mt_loader.get_loader("flood")
        tasks = mt_loader.available_tasks()
    """

    def __init__(
        self,
        loaders: Dict[str, DataLoader],
    ):
        """Initialize multi-task data loader.

        Args:
            loaders: Dictionary mapping task name to DataLoader
        """
        self.loaders = loaders
        self.iterators: Dict[str, Iterator] = {}

        # Initialize iterators
        for task, loader in loaders.items():
            self.iterators[task] = iter(loader)

    def get_loader(self, task: str) -> DataLoader:
        """Get DataLoader for a specific task.

        Args:
            task: Task name

        Returns:
            DataLoader for the task

        Raises:
            KeyError: If task not found
        """
        if task not in self.loaders:
            available = self.available_tasks()
            raise KeyError(
                f"Task '{task}' not found. Available: {available}"
            )

        return self.loaders[task]

    def get_iterator(self, task: str) -> Iterator:
        """Get iterator for a specific task.

        Args:
            task: Task name

        Returns:
            Iterator for the task
        """
        if task not in self.iterators:
            self.iterators[task] = iter(self.loaders[task])
        return self.iterators[task]

    def next_batch(self, task: str) -> Any:
        """Get next batch for a task.

        Args:
            task: Task name

        Returns:
            Next batch

        Raises:
            StopIteration: When loader is exhausted
        """
        iterator = self.get_iterator(task)

        try:
            return next(iterator)
        except StopIteration:
            # Reset iterator when exhausted
            self.iterators[task] = iter(self.loaders[task])
            return next(self.iterators[task])

    def available_tasks(self) -> List[str]:
        """Get list of available task names.

        Returns:
            List of task names
        """
        return list(self.loaders.keys())

    def task_count(self) -> int:
        """Get number of task loaders.

        Returns:
            Number of tasks
        """
        return len(self.loaders)

    def reset(self, task: Optional[str] = None) -> None:
        """Reset iterators for a task or all tasks.

        Args:
            task: Task to reset, or None for all
        """
        if task is None:
            for t in self.loaders:
                self.iterators[t] = iter(self.loaders[t])
        elif task in self.loaders:
            self.iterators[task] = iter(self.loaders[task])

    def get_batch_sizes(self) -> Dict[str, int]:
        """Get batch sizes for all loaders.

        Returns:
            Dictionary mapping task to batch size
        """
        return {
            task: loader.batch_size
            for task, loader in self.loaders.items()
        }

    def add_loader(self, task: str, loader: DataLoader) -> None:
        """Add a task loader.

        Args:
            task: Task name
            loader: DataLoader instance
        """
        self.loaders[task] = loader
        self.iterators[task] = iter(loader)

    def remove_loader(self, task: str) -> bool:
        """Remove a task loader.

        Args:
            task: Task name

        Returns:
            True if was present and now removed
        """
        if task in self.loaders:
            del self.loaders[task]
        if task in self.iterators:
            del self.iterators[task]
        return True

    def get_total_steps_per_epoch(self, steps_per_task: int) -> int:
        """Calculate total steps per epoch.

        Args:
            steps_per_task: Steps to run per task

        Returns:
            Total steps across all tasks
        """
        return steps_per_task * len(self.loaders)


class RoundRobinMultiTaskLoader(MultiTaskDataLoader):
    """Multi-task loader that cycles through tasks in round-robin.

    Provides balanced sampling across tasks.
    """

    def __init__(self, loaders: Dict[str, DataLoader]):
        """Initialize round-robin multi-task loader.

        Args:
            loaders: Dictionary mapping task name to DataLoader
        """
        super().__init__(loaders)
        self.current_task_idx = 0

    def next_round_robin(self) -> tuple:
        """Get next batch using round-robin.

        Returns:
            Tuple of (task_name, batch)
        """
        tasks = list(self.loaders.keys())
        task = tasks[self.current_task_idx]

        self.current_task_idx = (self.current_task_idx + 1) % len(tasks)

        batch = self.next_batch(task)

        return task, batch

    def reset(self, task: Optional[str] = None) -> None:
        """Reset iterators and index.

        Args:
            task: Task to reset, or None for all
        """
        super().reset(task)
        self.current_task_idx = 0


class RandomMultiTaskLoader(MultiTaskDataLoader):
    """Multi-task loader that randomly selects tasks.

    Provides weighted random sampling across tasks.
    """

    def __init__(
        self,
        loaders: Dict[str, DataLoader],
        weights: Optional[List[float]] = None,
    ):
        """Initialize random multi-task loader.

        Args:
            loaders: Dictionary mapping task name to DataLoader
            weights: Optional sampling weights
        """
        super().__init__(loaders)
        self.weights = weights or [1.0] * len(loaders)

    def next_random(self) -> tuple:
        """Get next batch using random sampling.

        Returns:
            Tuple of (task_name, batch)
        """
        import random

        tasks = list(self.loaders.keys())
        task = random.choices(tasks, weights=self.weights, k=1)[0]

        batch = self.next_batch(task)

        return task, batch