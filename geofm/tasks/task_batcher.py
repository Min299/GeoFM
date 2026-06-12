"""geofm.tasks.task_batcher

Task batcher for multi-task learning.

Provides utilities for managing and batching data from multiple tasks.
"""
from __future__ import annotations

from typing import Dict, List, Any, Optional, Iterator, Callable
from dataclasses import dataclass


@dataclass
class TaskBatch:
    """A batch of data for a specific task."""

    task_name: str
    inputs: Any
    targets: Any
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "task": self.task_name,
            "inputs": self.inputs,
            "targets": self.targets,
            "metadata": self.metadata or {},
        }


class TaskBatcher:
    """Batcher for managing data from multiple tasks.

    Usage:
        loaders = {
            "flood": flood_loader,
            "burn": burn_loader,
        }
        batcher = TaskBatcher(loaders)

        batch = batcher.get_batch("flood")  # Get flood batch
        batch = batcher.get_batch("burn")   # Get burn batch
    """

    def __init__(
        self,
        loaders: Dict[str, Any],
    ):
        """Initialize task batcher.

        Args:
            loaders: Dictionary mapping task name to data loader
        """
        self.loaders = loaders
        self.iterators: Dict[str, Iterator] = {}

        # Initialize iterators for each loader
        for task, loader in loaders.items():
            self.iterators[task] = iter(loader)

    def get_batch(self, task: str) -> TaskBatch:
        """Get next batch for a task.

        Args:
            task: Task name

        Returns:
            TaskBatch for the task

        Raises:
            KeyError: If task not in loaders
        """
        if task not in self.loaders:
            raise KeyError(f"Task '{task}' not found in loaders")

        try:
            batch = next(self.iterators[task])
        except StopIteration:
            # Reset iterator when exhausted
            self.iterators[task] = iter(self.loaders[task])
            batch = next(self.iterators[task])

        return TaskBatch(
            task_name=task,
            inputs=batch.get("inputs", batch.get("x")),
            targets=batch.get("targets", batch.get("y")),
            metadata=batch.get("metadata", {}),
        )

    def reset(self, task: Optional[str] = None) -> None:
        """Reset iterator for a task or all tasks.

        Args:
            task: Task to reset, or None for all tasks
        """
        if task is None:
            # Reset all iterators
            for t in self.loaders:
                self.iterators[t] = iter(self.loaders[t])
        elif task in self.loaders:
            self.iterators[task] = iter(self.loaders[task])

    def get_all_batches(self, task: str) -> List[TaskBatch]:
        """Get all batches for a task.

        Args:
            task: Task name

        Returns:
            List of TaskBatch objects
        """
        if task not in self.loaders:
            raise KeyError(f"Task '{task}' not found in loaders")

        batches = []
        for batch in self.loaders[task]:
            batches.append(TaskBatch(
                task_name=task,
                inputs=batch.get("inputs", batch.get("x")),
                targets=batch.get("targets", batch.get("y")),
                metadata=batch.get("metadata", {}),
            ))

        return batches

    def tasks(self) -> List[str]:
        """Get list of available tasks.

        Returns:
            List of task names
        """
        return list(self.loaders.keys())

    def add_loader(self, task: str, loader: Any) -> None:
        """Add a new task loader.

        Args:
            task: Task name
            loader: Data loader
        """
        self.loaders[task] = loader
        self.iterators[task] = iter(loader)

    def remove_loader(self, task: str) -> None:
        """Remove a task loader.

        Args:
            task: Task name
        """
        if task in self.loaders:
            del self.loaders[task]
        if task in self.iterators:
            del self.iterators[task]

    def loader_count(self) -> int:
        """Get number of loaders.

        Returns:
            Number of task loaders
        """
        return len(self.loaders)


class DynamicBatcher(TaskBatcher):
    """Batcher that creates batches on-the-fly.

    Useful for variable-sized inputs or custom batching logic.
    """

    def __init__(
        self,
        loaders: Dict[str, Any],
        batch_fn: Optional[Callable] = None,
    ):
        """Initialize dynamic batcher.

        Args:
            loaders: Dictionary mapping task name to data loader
            batch_fn: Optional custom batch creation function
        """
        super().__init__(loaders)
        self.batch_fn = batch_fn or self._default_batch_fn

    def _default_batch_fn(self, batch: Dict) -> TaskBatch:
        """Default batch creation function.

        Args:
            batch: Raw batch dictionary

        Returns:
            TaskBatch
        """
        return TaskBatch(
            task_name="unknown",
            inputs=batch.get("inputs", batch.get("x")),
            targets=batch.get("targets", batch.get("y")),
            metadata=batch.get("metadata", {}),
        )

    def get_batch_with_fn(self, task: str, batch_fn: Callable) -> TaskBatch:
        """Get batch using custom function.

        Args:
            task: Task name
            batch_fn: Custom batch function

        Returns:
            TaskBatch
        """
        raw_batch = self.get_raw_batch(task)
        return batch_fn(raw_batch)

    def get_raw_batch(self, task: str) -> Dict:
        """Get raw batch without TaskBatch wrapping.

        Args:
            task: Task name

        Returns:
            Raw batch dictionary
        """
        if task not in self.loaders:
            raise KeyError(f"Task '{task}' not found in loaders")

        try:
            batch = next(self.iterators[task])
        except StopIteration:
            self.iterators[task] = iter(self.loaders[task])
            batch = next(self.iterators[task])

        return batch


class MultiTaskBatch:
    """Container for multiple task batches.

    Holds batches from multiple tasks simultaneously.
    """

    def __init__(self):
        """Initialize multi-task batch container."""
        self.batches: Dict[str, TaskBatch] = {}

    def add(self, batch: TaskBatch) -> None:
        """Add a batch for a task.

        Args:
            batch: TaskBatch to add
        """
        self.batches[batch.task_name] = batch

    def get(self, task: str) -> Optional[TaskBatch]:
        """Get batch for a task.

        Args:
            task: Task name

        Returns:
            TaskBatch or None
        """
        return self.batches.get(task)

    def tasks(self) -> List[str]:
        """Get list of tasks in this multi-task batch.

        Returns:
            List of task names
        """
        return list(self.batches.keys())

    def count(self) -> int:
        """Get number of tasks in batch.

        Returns:
            Number of tasks
        """
        return len(self.batches)

    def to_dict(self) -> Dict:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            task: batch.to_dict()
            for task, batch in self.batches.items()
        }


class RoundRobinBatcher(TaskBatcher):
    """Batcher that cycles through tasks in round-robin."""

    def __init__(self, loaders: Dict[str, Any]):
        """Initialize round-robin batcher.

        Args:
            loaders: Dictionary mapping task name to data loader
        """
        super().__init__(loaders)
        self.current_task_index = 0

    def get_next_round_robin(self) -> TaskBatch:
        """Get next batch using round-robin scheduling.

        Returns:
            TaskBatch
        """
        tasks = list(self.loaders.keys())
        task = tasks[self.current_task_index]
        self.current_task_index = (self.current_task_index + 1) % len(tasks)

        return self.get_batch(task)

    def reset(self, task: Optional[str] = None) -> None:
        """Reset iterators and index.

        Args:
            task: Task to reset, or None for all
        """
        super().reset(task)
        self.current_task_index = 0