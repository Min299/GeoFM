"""geofm.benchmark.shared_vs_independent

Benchmark for comparing SharedGeoFM vs SeparateGeoFM models.
"""
from __future__ import annotations

import torch


def count_parameters(
    model,
):
    """Count total parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Total number of parameters
    """
    return sum(
        p.numel()
        for p in model.parameters()
    )


def benchmark_models(
    shared_model,
    independent_model,
):
    """Benchmark and compare shared vs independent models.

    Args:
        shared_model: SharedGeoFM model
        independent_model: SeparateGeoFM model

    Returns:
        Dictionary with parameter counts for both models
    """
    return {
        "shared_parameters":
            count_parameters(
                shared_model
            ),
        "independent_parameters":
            count_parameters(
                independent_model
            ),
    }