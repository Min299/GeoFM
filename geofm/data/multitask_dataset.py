"""geofm.data.multitask_dataset

Multi-task dataset that combines multiple task datasets.

Provides a unified interface for training on multiple tasks.
"""
from __future__ import annotations

from typing import Dict, Any, List
from torch.utils.data import Dataset


class MultiTaskDataset(Dataset):
    """Dataset that combines multiple task datasets.

    Provides a unified interface for multi-task learning by combining
    multiple task-specific datasets into a single dataset.

    Usage:
        datasets = {
            "flood": flood_dataset,
            "burn": burn_dataset,
        }

        mt_dataset = MultiTaskDataset(datasets)

        sample = mt_dataset[0]  # Returns sample with "task" field
    """

    def __init__(
        self,
        task_datasets: Dict[str, Dataset],
        cycle: bool = True,
    ):
        """Initialize multi-task dataset.

        Args:
            task_datasets: Dictionary mapping task name to dataset
            cycle: If True, cycle through tasks evenly; if False, concatenate
        """
        self.task_datasets = task_datasets
        self.tasks = list(task_datasets.keys())
        self.cycle = cycle

    def __len__(self) -> int:
        """Get total number of samples.

        Returns:
            Total sample count (minimum across tasks if cycling)
        """
        if self.cycle:
            # Cycle through all tasks evenly
            return min(len(ds) for ds in self.task_datasets.values()) * len(self.tasks)
        else:
            # Concatenate all samples
            return sum(len(ds) for ds in self.task_datasets.values())

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get sample by index.

        When cycling, distributes indices evenly across tasks.
        When concatenating, maps index to appropriate task.

        Args:
            index: Sample index

        Returns:
            Sample dictionary with "task" field
        """
        if self.cycle:
            return self._get_item_cyclic(index)
        else:
            return self._get_item_concat(index)

    def _get_item_cyclic(self, index: int) -> Dict[str, Any]:
        """Get item using cyclic distribution.

        Args:
            index: Sample index

        Returns:
            Sample dictionary
        """
        num_tasks = len(self.tasks)
        task_idx = index % num_tasks
        sample_idx = index // num_tasks

        task = self.tasks[task_idx]
        dataset = self.task_datasets[task]

        sample = dataset[sample_idx % len(dataset)]
        sample["task"] = task

        return sample

    def _get_item_concat(self, index: int) -> Dict[str, Any]:
        """Get item using concatenation.

        Args:
            index: Sample index

        Returns:
            Sample dictionary
        """
        cumsum = 0
        for task, dataset in self.task_datasets.items():
            if index < cumsum + len(dataset):
                sample = dataset[index - cumsum]
                sample["task"] = task
                return sample
            cumsum += len(dataset)

        # Wrap around
        sample = list(self.task_datasets.values())[0][0]
        sample["task"] = self.tasks[0]
        return sample

    def get_task_dataset(self, task: str) -> Dataset:
        """Get dataset for a specific task.

        Args:
            task: Task name

        Returns:
            Dataset for the task
        """
        if task not in self.task_datasets:
            raise KeyError(f"Task '{task}' not found")
        return self.task_datasets[task]

    def get_task_indices(self, task: str) -> List[int]:
        """Get indices for a specific task.

        Args:
            task: Task name

        Returns:
            List of indices that map to this task
        """
        if task not in self.task_datasets:
            raise KeyError(f"Task '{task}' not found")

        if self.cycle:
            task_idx = self.tasks.index(task)
            min_len = min(len(ds) for ds in self.task_datasets.values())

            return [
                i * len(self.tasks) + task_idx
                for i in range(min_len)
            ]
        else:
            # Calculate concatenated indices
            indices = []
            cumsum = 0
            for t, ds in self.task_datasets.items():
                if t == task:
                    return list(range(cumsum, cumsum + len(ds)))
                cumsum += len(ds)
            return []

    def available_tasks(self) -> List[str]:
        """Get list of available tasks.

        Returns:
            List of task names
        """
        return self.tasks

    def task_count(self) -> int:
        """Get number of tasks.

        Returns:
            Number of tasks
        """
        return len(self.tasks)

    def get_task_sample_counts(self) -> Dict[str, int]:
        """Get sample count for each task.

        Returns:
            Dictionary mapping task to sample count
        """
        return {
            task: len(dataset)
            for task, dataset in self.task_datasets.items()
        }


class ConcatMultiTaskDataset(MultiTaskDataset):
    """Multi-task dataset that concatenates all datasets.

    Unlike MultiTaskDataset which cycles, this concatenates
    all samples sequentially.
    """

    def __init__(self, task_datasets: Dict[str, Dataset]):
        """Initialize concat multi-task dataset.

        Args:
            task_datasets: Dictionary mapping task name to dataset
        """
        super().__init__(task_datasets, cycle=False)


class CycleMultiTaskDataset(MultiTaskDataset):
    """Multi-task dataset that cycles evenly through tasks.

    Ensures each task contributes equally to each epoch.
    """

    def __init__(self, task_datasets: Dict[str, Dataset]):
        """Initialize cycle multi-task dataset.

        Args:
            task_datasets: Dictionary mapping task name to dataset
        """
        super().__init__(task_datasets, cycle=True)