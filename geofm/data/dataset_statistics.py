"""geofm.data.dataset_statistics

Utilities for computing and analyzing dataset statistics.

Provides statistical analysis of datasets.
"""
from __future__ import annotations

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import torch


@dataclass
class DatasetStats:
    """Statistics for a dataset."""

    num_samples: int
    mean: float
    std: float
    min: float
    max: float


class DatasetStatistics:
    """Utilities for computing dataset statistics."""

    @staticmethod
    def compute_mean_std(
        dataset,
        num_samples: int = 100,
        key: str = "image",
    ) -> Dict[str, float]:
        """Compute mean and standard deviation of dataset images.

        Args:
            dataset: PyTorch dataset
            num_samples: Number of samples to compute statistics over
            key: Key in sample dict containing the tensor

        Returns:
            Dictionary with "mean" and "std" values
        """
        values = []

        limit = min(len(dataset), num_samples)

        for idx in range(limit):
            sample = dataset[idx]
            tensor = sample[key]

            if torch.is_tensor(tensor):
                values.append(tensor.float())

        if not values:
            return {"mean": 0.0, "std": 0.0}

        stacked = torch.stack(values)

        mean = stacked.mean()
        std = stacked.std()

        return {
            "mean": float(mean),
            "std": float(std),
        }

    @staticmethod
    def compute_batch_stats(
        batch: Dict[str, Any],
        keys: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Compute statistics for a batch.

        Args:
            batch: Batch dictionary
            keys: Optional list of keys to compute stats for

        Returns:
            Dictionary mapping key to stats dict
        """
        if keys is None:
            keys = list(batch.keys())

        stats = {}

        for key in keys:
            value = batch[key]

            if torch.is_tensor(value):
                stats[key] = {
                    "mean": float(value.mean()),
                    "std": float(value.std()),
                    "min": float(value.min()),
                    "max": float(value.max()),
                    "shape": tuple(value.shape),
                }

        return stats

    @staticmethod
    def get_dataset_info(
        dataset,
        sample_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Get general information about a dataset.

        Args:
            dataset: PyTorch dataset
            sample_keys: Keys to inspect in first sample

        Returns:
            Dictionary with dataset information
        """
        info = {
            "length": len(dataset),
        }

        if sample_keys:
            try:
                sample = dataset[0]
                for key in sample_keys:
                    if key in sample:
                        value = sample[key]
                        if hasattr(value, "shape"):
                            info[f"{key}_shape"] = tuple(value.shape)
                        else:
                            info[f"{key}_type"] = type(value).__name__
            except Exception:
                pass

        return info

    @staticmethod
    def analyze_class_distribution(
        dataset,
        label_key: str = "label",
    ) -> Dict[int, int]:
        """Analyze class distribution in dataset.

        Args:
            dataset: PyTorch dataset
            label_key: Key containing class labels

        Returns:
            Dictionary mapping class to count
        """
        distribution = {}

        for idx in range(len(dataset)):
            sample = dataset[idx]
            label = sample.get(label_key)

            if label is not None:
                if torch.is_tensor(label):
                    label = label.item()

                distribution[label] = distribution.get(label, 0) + 1

        return distribution

    @staticmethod
    def compute_sample_ratios(
        dataset,
        task_key: str = "task",
    ) -> Dict[str, float]:
        """Compute sampling ratios for multi-task dataset.

        Args:
            dataset: Multi-task dataset
            task_key: Key containing task name

        Returns:
            Dictionary mapping task to ratio
        """
        total = len(dataset)
        task_counts = {}

        for idx in range(total):
            sample = dataset[idx]
            task = sample.get(task_key)

            if task:
                task_counts[task] = task_counts.get(task, 0) + 1

        return {
            task: count / total
            for task, count in task_counts.items()
        }