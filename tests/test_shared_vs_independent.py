"""tests/test_shared_vs_independent.py

Tests for shared_vs_independent benchmark.
"""
import torch.nn as nn

from geofm.benchmark.shared_vs_independent import (
    benchmark_models,
    count_parameters,
)


class Dummy(nn.Module):
    """Dummy model for testing."""

    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(
            10,
            10,
        )

    def forward(
        self,
        x,
    ):
        return x


def test_count_parameters():
    """Test parameter counting."""
    model = Dummy()
    count = count_parameters(model)

    assert count > 0
    assert isinstance(count, int)


def test_benchmark():
    """Test benchmark comparison."""
    results = benchmark_models(
        Dummy(),
        Dummy(),
    )

    assert (
        "shared_parameters"
        in results
    )

    assert (
        "independent_parameters"
        in results
    )


def test_benchmark_returns_ints():
    """Test that benchmark returns integer counts."""
    results = benchmark_models(
        Dummy(),
        Dummy(),
    )

    assert isinstance(results["shared_parameters"], int)
    assert isinstance(results["independent_parameters"], int)