"""geofm.data.sample_visualizer

Utilities for visualizing and inspecting data samples.

Provides tools for debugging and understanding dataset samples.
"""
from __future__ import annotations

from typing import Dict, Any, List, Optional
import torch


class SampleVisualizer:
    """Utilities for visualizing and inspecting samples."""

    @staticmethod
    def summarize(sample: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize a sample's structure.

        Args:
            sample: Sample dictionary

        Returns:
            Dictionary with shape/type info for each key
        """
        summary = {}

        for key, value in sample.items():
            if hasattr(value, "shape"):
                summary[key] = {
                    "shape": tuple(value.shape),
                    "dtype": str(value.dtype) if torch.is_tensor(value) else type(value).__name__,
                    "type": "tensor" if torch.is_tensor(value) else type(value).__name__,
                }
            else:
                summary[key] = {
                    "type": type(value).__name__,
                    "value": str(value)[:100],  # Truncate long values
                }

        return summary

    @staticmethod
    def print_sample(sample: Dict[str, Any], indent: int = 2) -> None:
        """Pretty print a sample.

        Args:
            sample: Sample dictionary
            indent: Indentation level
        """
        print("Sample:")
        print("-" * 40)

        for key, value in sample.items():
            if hasattr(value, "shape"):
                shape = tuple(value.shape)
                dtype = str(value.dtype) if torch.is_tensor(value) else ""
                print(f"  {key}: shape={shape}, dtype={dtype}")
            else:
                print(f"  {key}: {type(value).__name__} = {value}")

        print("-" * 40)

    @staticmethod
    def validate_sample(
        sample: Dict[str, Any],
        required_keys: Optional[List[str]] = None,
    ) -> Dict[str, bool]:
        """Validate that a sample has expected structure.

        Args:
            sample: Sample dictionary
            required_keys: Optional list of required keys

        Returns:
            Dictionary with validation results
        """
        results = {}

        # Check required keys
        if required_keys:
            for key in required_keys:
                results[f"has_{key}"] = key in sample

        # Check tensor shapes
        for key, value in sample.items():
            if torch.is_tensor(value):
                results[f"{key}_is_tensor"] = True
                results[f"{key}_is_1d"] = len(value.shape) == 1
                results[f"{key}_is_2d"] = len(value.shape) == 2
                results[f"{key}_is_3d"] = len(value.shape) == 3
                results[f"{key}_is_4d"] = len(value.shape) == 4
            else:
                results[f"{key}_is_tensor"] = False

        return results

    @staticmethod
    def compare_samples(
        sample1: Dict[str, Any],
        sample2: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compare two samples.

        Args:
            sample1: First sample
            sample2: Second sample

        Returns:
            Dictionary with comparison results
        """
        comparison = {
            "same_keys": set(sample1.keys()) == set(sample2.keys()),
            "keys_in_both": list(set(sample1.keys()) & set(sample2.keys())),
            "keys_only_in_first": list(set(sample1.keys()) - set(sample2.keys())),
            "keys_only_in_second": list(set(sample2.keys()) - set(sample1.keys())),
        }

        # Compare tensor values where applicable
        for key in comparison["keys_in_both"]:
            v1, v2 = sample1[key], sample2[key]
            if torch.is_tensor(v1) and torch.is_tensor(v2):
                comparison[f"{key}_same_shape"] = v1.shape == v2.shape
                comparison[f"{key}_same_dtype"] = v1.dtype == v2.dtype

        return comparison

    @staticmethod
    def get_tensor_info(tensor: torch.Tensor) -> Dict[str, Any]:
        """Get detailed information about a tensor.

        Args:
            tensor: PyTorch tensor

        Returns:
            Dictionary with tensor information
        """
        return {
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "requires_grad": tensor.requires_grad,
            "min": float(tensor.min()) if tensor.numel() > 0 else None,
            "max": float(tensor.max()) if tensor.numel() > 0 else None,
            "mean": float(tensor.mean()) if tensor.numel() > 0 else None,
            "std": float(tensor.std()) if tensor.numel() > 0 else None,
            "numel": tensor.numel(),
        }

    @staticmethod
    def describe_batch(batch: Dict[str, Any]) -> str:
        """Get a text description of a batch.

        Args:
            batch: Batch dictionary

        Returns:
            String description
        """
        lines = ["Batch Description:"]

        for key, value in batch.items():
            if torch.is_tensor(value):
                lines.append(f"  {key}: tensor shape={tuple(value.shape)}, dtype={value.dtype}")
            elif isinstance(value, list):
                lines.append(f"  {key}: list of {len(value)} items")
            elif isinstance(value, dict):
                lines.append(f"  {key}: dict with {len(value)} keys")
            else:
                lines.append(f"  {key}: {type(value).__name__}")

        return "\n".join(lines)