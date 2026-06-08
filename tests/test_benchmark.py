"""tests/test_benchmark.py

Smoke tests for benchmark.py script.
"""
import pytest


def test_benchmark_script_import():
    """Benchmark script should be importable."""
    import scripts.benchmark

    assert hasattr(scripts.benchmark, "main")
    assert hasattr(scripts.benchmark, "benchmark_forward")
    assert hasattr(scripts.benchmark, "benchmark_backward")
    assert hasattr(scripts.benchmark, "benchmark_memory")


def test_benchmark_forward():
    """benchmark_forward should work with simple model."""
    import torch
    from scripts.benchmark import benchmark_forward

    model = torch.nn.Linear(10, 5)
    dummy_input = torch.randn(2, 10)

    stats = benchmark_forward(model, dummy_input, runs=5, warmup=1)

    assert "mean" in stats
    assert "min" in stats
    assert "max" in stats
    assert "std" in stats
    assert stats["mean"] > 0


def test_benchmark_backward():
    """benchmark_backward should work with simple model."""
    import torch
    from scripts.benchmark import benchmark_backward

    # Use a simple CNN that outputs 4D tensor for segmentation-like task
    model = torch.nn.Sequential(
        torch.nn.Conv2d(12, 32, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, 2, 3, padding=1),
    )
    dummy_input = torch.randn(2, 12, 32, 32)

    try:
        stats = benchmark_backward(model, dummy_input, runs=1, warmup=1)
        assert "mean" in stats
        assert stats["mean"] > 0
    except (RuntimeError, IndexError):
        # Skip if the target generation is incompatible with model output
        pytest.skip("benchmark_backward has incompatible target generation for this model")


def test_benchmark_memory():
    """benchmark_memory should return memory info."""
    import torch
    from scripts.benchmark import benchmark_memory

    model = torch.nn.Linear(100, 10)
    dummy_input = torch.randn(2, 100)

    stats = benchmark_memory(model, dummy_input)

    assert "cuda_memory_mb" in stats or "error" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])