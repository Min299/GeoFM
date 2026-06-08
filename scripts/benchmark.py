#!/usr/bin/env python3
"""scripts/benchmark.py

Benchmark GeoFM model performance.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --adaptation lora --batch_size 4
"""
import argparse
import sys
import time
from pathlib import Path

import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from geofm.experiments import ExperimentConfig
from geofm.experiments.experiment_builder import ExperimentBuilder


def benchmark_forward(model, dummy_input, runs: int = 20, warmup: int = 5):
    """Benchmark forward pass."""
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(runs):
            start = time.perf_counter()
            _ = model(dummy_input)
            end = time.perf_counter()
            times.append(end - start)

    return {
        "mean": sum(times) / len(times),
        "min": min(times),
        "max": max(times),
        "std": (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
    }


def benchmark_backward(model, dummy_input, runs: int = 10, warmup: int = 3):
    """Benchmark forward + backward pass."""
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # Warmup
    for _ in range(warmup):
        optimizer.zero_grad()
        output = model(dummy_input)
        if isinstance(output, (list, tuple)):
            output = output[0]
        target = torch.randint(0, 2, output.shape[:2] + (output.shape[2], output.shape[3]))
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    # Benchmark
    times = []
    for _ in range(runs):
        start = time.perf_counter()

        optimizer.zero_grad()
        output = model(dummy_input)
        if isinstance(output, (list, tuple)):
            output = output[0]
        target = torch.randint(0, 2, output.shape[:2] + (output.shape[2], output.shape[3]))
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        end = time.perf_counter()
        times.append(end - start)

    return {
        "mean": sum(times) / len(times),
        "min": min(times),
        "max": max(times),
        "std": (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
    }


def benchmark_memory(model, dummy_input):
    """Estimate memory usage."""
    if not torch.cuda.is_available():
        return {"cuda_memory_mb": 0, "error": "CUDA not available"}

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    model = model.cuda()
    dummy_input = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in dummy_input.items()}

    with torch.no_grad():
        _ = model(dummy_input)

    memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB

    # Free memory
    model = model.cpu()
    torch.cuda.empty_cache()

    return {"cuda_memory_mb": memory_allocated}


def main():
    parser = argparse.ArgumentParser(description="Benchmark GeoFM model")

    parser.add_argument(
        "--task",
        type=str,
        default="flood",
        choices=["flood", "burn", "lulc"],
        help="Task type",
    )
    parser.add_argument(
        "--adaptation",
        type=str,
        default="lora",
        choices=["feature", "lora", "hybrid", "full_ft"],
        help="Adaptation method",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for benchmarking",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=20,
        help="Number of benchmark runs",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("GEOFM BENCHMARK")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Adaptation: {args.adaptation}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Runs: {args.runs}")
    print()

    # Create config
    config = ExperimentConfig(
        name=f"benchmark_{args.adaptation}",
        task=args.task,
        adaptation=args.adaptation,
        backbone="terramind_base",
    )

    # Build model
    print("Building model...")
    model = ExperimentBuilder.build(config)

    if model is None:
        print("Error: Model build returned None")
        return

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total:,} total, {trainable:,} trainable")

    # Create dummy input
    dummy_input = {
        "S2L1C": {
            "x": torch.randn(args.batch_size, 12, 224, 224)
        }
    }

    # Forward benchmark
    print()
    print("-" * 60)
    print("FORWARD PASS BENCHMARK")
    print("-" * 60)

    forward_stats = benchmark_forward(model, dummy_input, runs=args.runs)
    print(f"Mean:   {forward_stats['mean']*1000:.2f} ms")
    print(f"Min:    {forward_stats['min']*1000:.2f} ms")
    print(f"Max:    {forward_stats['max']*1000:.2f} ms")
    print(f"Std:    {forward_stats['std']*1000:.2f} ms")

    # Backward benchmark
    print()
    print("-" * 60)
    print("TRAINING (FORWARD + BACKWARD) BENCHMARK")
    print("-" * 60)

    backward_stats = benchmark_backward(model, dummy_input, runs=args.runs // 2)
    print(f"Mean:   {backward_stats['mean']*1000:.2f} ms")
    print(f"Min:    {backward_stats['min']*1000:.2f} ms")
    print(f"Max:    {backward_stats['max']*1000:.2f} ms")
    print(f"Std:    {backward_stats['std']*1000:.2f} ms")

    # Memory benchmark
    print()
    print("-" * 60)
    print("MEMORY USAGE")
    print("-" * 60)

    memory_stats = benchmark_memory(model, dummy_input)
    if "error" in memory_stats:
        print(memory_stats["error"])
    else:
        print(f"CUDA Memory: {memory_stats['cuda_memory_mb']:.1f} MB")

    print()
    print("=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()