"""tests/test_adapter_benchmark.py

Tests for adapter benchmark module.
"""
import pytest

from geofm.benchmark.adapter_benchmark import (
    AdapterBenchmark,
    BenchmarkResult,
)


class TestBenchmarkResult:
    """Test BenchmarkResult dataclass."""

    def test_create_result(self):
        """Creating a result should store values."""
        result = BenchmarkResult(
            adapter_type="feature",
            trainable_params=100000,
            total_params=85000000,
            forward_time=0.03,
            backward_time=0.05,
            memory_mb=2200,
            metric=0.75,
            metric_name="iou",
        )

        assert result.adapter_type == "feature"
        assert result.trainable_params == 100000
        assert result.metric == 0.75

    def test_trainable_percentage(self):
        """trainable_percentage should compute correctly."""
        result = BenchmarkResult(
            adapter_type="feature",
            trainable_params=100000,
            total_params=1000000,
            forward_time=0.03,
            backward_time=0.05,
            memory_mb=2200,
        )

        assert result.trainable_percentage == 10.0

    def test_trainable_percentage_zero_total(self):
        """trainable_percentage should handle zero total."""
        result = BenchmarkResult(
            adapter_type="feature",
            trainable_params=100,
            total_params=0,
            forward_time=0.03,
            backward_time=0.05,
            memory_mb=100,
        )

        assert result.trainable_percentage == 0.0


class TestAdapterBenchmark:
    """Test AdapterBenchmark class."""

    def test_add_result(self):
        """Adding result should store it."""
        bench = AdapterBenchmark()

        result = BenchmarkResult(
            adapter_type="feature",
            trainable_params=100000,
            total_params=85000000,
            forward_time=0.03,
            backward_time=0.05,
            memory_mb=2200,
        )

        bench.add_result(result)

        assert len(bench.results) == 1
        assert bench.results[0].adapter_type == "feature"

    def test_get_result(self):
        """Getting result should return correct adapter."""
        bench = AdapterBenchmark()

        bench.add_result(BenchmarkResult(
            adapter_type="feature",
            trainable_params=100000,
            total_params=85000000,
            forward_time=0.03,
            backward_time=0.05,
            memory_mb=2200,
        ))

        bench.add_result(BenchmarkResult(
            adapter_type="lora",
            trainable_params=17000000,
            total_params=85000000,
            forward_time=0.04,
            backward_time=0.06,
            memory_mb=2400,
        ))

        result = bench.get_result("lora")
        assert result is not None
        assert result.adapter_type == "lora"

    def test_get_result_not_found(self):
        """Getting non-existent result should return None."""
        bench = AdapterBenchmark()

        result = bench.get_result("nonexistent")
        assert result is None

    def test_summary(self):
        """Summary should return list of dicts."""
        bench = AdapterBenchmark()

        bench.add_result(BenchmarkResult(
            adapter_type="feature",
            trainable_params=100000,
            total_params=85000000,
            forward_time=0.03,
            backward_time=0.05,
            memory_mb=2200,
            metric=0.75,
            metric_name="iou",
        ))

        summary = bench.summary()

        assert len(summary) == 1
        assert summary[0]["adapter"] == "feature"
        assert summary[0]["trainable_params"] == 100000
        assert "iou" in summary[0]

    def test_clear(self):
        """Clear should remove all results."""
        bench = AdapterBenchmark()

        bench.add_result(BenchmarkResult(
            adapter_type="feature",
            trainable_params=100000,
            total_params=85000000,
            forward_time=0.03,
            backward_time=0.05,
            memory_mb=2200,
        ))

        assert len(bench.results) == 1

        bench.clear()

        assert len(bench.results) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])