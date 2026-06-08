"""geofm.data.dataloader_factory

Factory for creating DataLoaders.

Provides consistent interface for creating data loaders.
"""
from __future__ import annotations

from typing import Optional, Any, Callable
from torch.utils.data import DataLoader, Dataset

from geofm.data.collate import geofm_collate


class DataLoaderFactory:
    """Factory for creating DataLoaders with consistent settings.

    Usage:
        loader = DataLoaderFactory.build(
            dataset=train_dataset,
            batch_size=8,
            shuffle=True,
        )
    """

    @staticmethod
    def build(
        dataset: Dataset,
        batch_size: int = 8,
        shuffle: bool = True,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        **kwargs,
    ) -> DataLoader:
        """Build a DataLoader with standard settings.

        Args:
            dataset: PyTorch dataset
            batch_size: Batch size
            shuffle: Whether to shuffle
            num_workers: Number of worker processes
            collate_fn: Optional custom collate function
            pin_memory: Whether to pin memory
            drop_last: Whether to drop last incomplete batch
            **kwargs: Additional DataLoader arguments

        Returns:
            DataLoader instance
        """
        if collate_fn is None:
            collate_fn = geofm_collate

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            **kwargs,
        )

    @staticmethod
    def build_train_loader(
        dataset: Dataset,
        batch_size: int = 8,
        num_workers: int = 0,
        **kwargs,
    ) -> DataLoader:
        """Build a training DataLoader.

        Args:
            dataset: PyTorch dataset
            batch_size: Batch size
            num_workers: Number of workers
            **kwargs: Additional arguments

        Returns:
            Training DataLoader
        """
        return DataLoaderFactory.build(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            **kwargs,
        )

    @staticmethod
    def build_val_loader(
        dataset: Dataset,
        batch_size: int = 8,
        num_workers: int = 0,
        **kwargs,
    ) -> DataLoader:
        """Build a validation DataLoader.

        Args:
            dataset: PyTorch dataset
            batch_size: Batch size
            num_workers: Number of workers
            **kwargs: Additional arguments

        Returns:
            Validation DataLoader
        """
        return DataLoaderFactory.build(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            **kwargs,
        )

    @staticmethod
    def build_test_loader(
        dataset: Dataset,
        batch_size: int = 8,
        num_workers: int = 0,
        **kwargs,
    ) -> DataLoader:
        """Build a test DataLoader.

        Args:
            dataset: PyTorch dataset
            batch_size: Batch size
            num_workers: Number of workers
            **kwargs: Additional arguments

        Returns:
            Test DataLoader
        """
        return DataLoaderFactory.build(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            **kwargs,
        )


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Convenience function for creating a DataLoader.

    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of workers

    Returns:
        DataLoader instance
    """
    return DataLoaderFactory.build(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )