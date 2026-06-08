"""geofm.benchmark.adapter_benchmark

Benchmark results for adapter comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    adapter_type: str
    trainable_params: int
    total_params: int
    forward_time: float
    backward_time: float
    memory_mb: float
    metric: Optional[float] = None
    metric_name: str = "iou"

    @property
    def trainable_percentage(self) -> float:
        """Percentage of trainable parameters."""
        if self.total_params == 0:
            return 0.0
        return 100 * self.trainable_params / self.total_params


@dataclass
class AdapterBenchmark:
    """Benchmark runner for adapter comparison."""

    results: list[BenchmarkResult] = field(default_factory=list)
    name: str = "adapter_benchmark"

    def add_result(self, result: BenchmarkResult) -> None:
        """Add a benchmark result.

        Args:
            result: Benchmark result to add
        """
        self.results.append(result)

    def get_result(self, adapter_type: str) -> Optional[BenchmarkResult]:
        """Get result for specific adapter type.

        Args:
            adapter_type: Type of adapter (feature, lora, hybrid, full_ft)

        Returns:
            Benchmark result or None if not found
        """
        for r in self.results:
            if r.adapter_type == adapter_type:
                return r
        return None

    def summary(self) -> list[dict]:
        """Get summary of all results.

        Returns:
            List of result dictionaries
        """
        rows = []
        for r in self.results:
            rows.append({
                "adapter": r.adapter_type,
                "trainable_params": r.trainable_params,
                "total_params": r.total_params,
                "trainable_%": f"{r.trainable_percentage:.2f}%",
                "forward_ms": f"{r.forward_time * 1000:.2f}",
                "backward_ms": f"{r.backward_time * 1000:.2f}",
                "memory_mb": f"{r.memory_mb:.0f}",
                r.metric_name: f"{r.metric:.4f}" if r.metric else "N/A",
            })
        return rows

    def print_summary(self) -> None:
        """Print formatted summary table."""
        print("=" * 80)
        print(f"BENCHMARK: {self.name}")
        print("=" * 80)

        # Header
        print(f"{'Adapter':<12} {'Params':>12} {'Train%':>8} {'Fwd ms':>10} {'Bwd ms':>10} {'Mem MB':>10}")
        print("-" * 80)

        # Rows
        for r in self.results:
            print(
                f"{r.adapter_type:<12} "
                f"{r.trainable_params:>12,} "
                f"{r.trainable_percentage:>7.2f}% "
                f"{r.forward_time*1000:>9.2f} "
                f"{r.backward_time*1000:>9.2f} "
                f"{r.memory_mb:>9.0f}"
            )

        print("=" * 80)

    def clear(self) -> None:
        """Clear all results."""
        self.results.clear()