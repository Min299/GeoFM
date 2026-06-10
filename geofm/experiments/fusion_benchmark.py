"""geofm.experiments.fusion_benchmark

Benchmark utilities for fusion modules.
"""
from __future__ import annotations

import time


def benchmark_fusion(
    fusion,
    image_tokens,
    metadata_tokens,
):
    """Benchmark fusion module latency.

    Args:
        fusion: Fusion module to benchmark
        image_tokens: Image token tensor
        metadata_tokens: Metadata token tensor

    Returns:
        Dictionary with latency_ms metric
    """
    start = time.time()

    fusion(
        image_tokens,
        metadata_tokens,
    )

    end = time.time()

    return {
        "latency_ms":
            (end - start)
            * 1000
    }