"""tests/test_fusion_benchmark.py

Tests for fusion benchmark.
"""
import torch

from geofm.models.fusion.concat_fusion import (
    ConcatFusion,
)

from geofm.experiments.fusion_benchmark import (
    benchmark_fusion,
)


def test_benchmark():
    """Test basic benchmark functionality."""
    fusion = ConcatFusion()

    img = torch.randn(
        2,
        196,
        128,
    )

    meta = torch.randn(
        2,
        1,
        128,
    )

    result = (
        benchmark_fusion(
            fusion,
            img,
            meta,
        )
    )

    assert (
        "latency_ms"
        in result
    )


def test_benchmark_returns_float():
    """Test that benchmark returns a positive latency."""
    fusion = ConcatFusion()

    img = torch.randn(1, 10, 64)
    meta = torch.randn(1, 1, 64)

    result = benchmark_fusion(fusion, img, meta)

    assert isinstance(result["latency_ms"], float)
    assert result["latency_ms"] >= 0