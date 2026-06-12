"""geofm.data.collate

Collate functions for batching data.

Provides custom collate functions for GeoFM datasets.
"""
from __future__ import annotations

from typing import Dict, List, Any
import torch


def geofm_collate(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for GeoFM batches.

    Handles both tensor and non-tensor values appropriately.

    Args:
        batch: List of sample dictionaries

    Returns:
        Batched dictionary
    """
    result = {}

    if not batch:
        return result

    keys = batch[0].keys()

    for key in keys:
        values = [item[key] for item in batch]

        if torch.is_tensor(values[0]):
            # Stack tensors
            result[key] = torch.stack(values, dim=0)
        elif all(isinstance(v, (int, float)) for v in values):
            # Stack scalars as tensor
            result[key] = torch.tensor(values)
        elif all(isinstance(v, str) for v in values):
            # Keep strings as list
            result[key] = values
        elif all(isinstance(v, list) for v in values):
            # Keep lists as list of lists
            result[key] = values
        elif all(isinstance(v, dict) for v in values):
            # Recursively collate dicts
            result[key] = [geofm_collate([v]) for v in values]
        else:
            # Default: keep as list
            result[key] = values

    return result


def segmentation_collate(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for segmentation tasks.

    Handles images and masks specifically.

    Args:
        batch: List of sample dictionaries

    Returns:
        Batched dictionary
    """
    result = {}

    for key in ["image", "mask", "inputs", "targets"]:
        if key in batch[0]:
            values = [item[key] for item in batch]

            if torch.is_tensor(values[0]):
                result[key] = torch.stack(values, dim=0)
            else:
                result[key] = values

    # Handle metadata
    for key in ["metadata", "sample_id", "path"]:
        if key in batch[0]:
            result[key] = [item[key] for item in batch]

    return result


def multitask_collate(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for multi-task batches.

    Ensures task information is preserved.

    Args:
        batch: List of sample dictionaries

    Returns:
        Batched dictionary
    """
    result = geofm_collate(batch)

    # Ensure task is always a list
    if "task" in batch[0]:
        result["task"] = [item["task"] for item in batch]

    return result


def variable_size_collate(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for variable-size inputs.

    Pads sequences to the same length.

    Args:
        batch: List of sample dictionaries

    Returns:
        Batched dictionary with padding
    """
    result = {}

    for key in batch[0].keys():
        values = [item[key] for item in batch]

        if torch.is_tensor(values[0]):
            # For variable size, pad to max
            max_size = max(v.shape[0] for v in values)
            padded = []

            for v in values:
                if v.shape[0] < max_size:
                    # Pad
                    pad = torch.zeros(max_size - v.shape[0], *v.shape[1:])
                    v = torch.cat([v, pad], dim=0)
                padded.append(v)

            result[key] = torch.stack(padded, dim=0)
        else:
            result[key] = values

    return result


def ignore_none_collate(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function that filters out None values.

    Args:
        batch: List of sample dictionaries

    Returns:
        Batched dictionary without None values
    """
    # Filter out None samples
    filtered = [item for item in batch if item is not None]

    if not filtered:
        return {}

    return geofm_collate(filtered)