"""geofm.benchmark.benchmark_runner

Runner for adapter benchmarks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from geofm.benchmark.adapter_benchmark import (
    AdapterBenchmark,
    BenchmarkResult,
)
from geofm.experiments import ExperimentConfig
from geofm.experiments.experiment_builder import ExperimentBuilder


class BenchmarkRunner:
    """Runner that executes benchmarks for all adapter types.

    Usage:
        runner = BenchmarkRunner()
        results = runner.run_all()
        runner.benchmark.print_summary()
    """

    def __init__(
        self,
        batch_size: int = 2,
        num_runs: int = 20,
        warmup: int = 5,
        input_size: int = 224,
    ):
        """Initialize benchmark runner.

        Args:
            batch_size: Batch size for benchmarking
            num_runs: Number of benchmark iterations
            warmup: Number of warmup iterations
            input_size: Input image size
        """
        self.batch_size = batch_size
        self.num_runs = num_runs
        self.warmup = warmup
        self.input_size = input_size
        self.benchmark = AdapterBenchmark(name="adapter_comparison")

    def _create_dummy_input(self) -> dict:
        """Create dummy input for benchmarking."""
        return {
            "S2L1C": {
                "x": torch.randn(
                    self.batch_size,
                    12,  # Sentinel-2 has 12 bands
                    self.input_size,
                    self.input_size,
                )
            }
        }

    def _count_params(self, model: torch.nn.Module) -> tuple[int, int]:
        """Count total and trainable parameters.

        Returns:
            Tuple of (total_params, trainable_params)
        """
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable

    def _benchmark_forward(self, model: torch.nn.Module) -> float:
        """Benchmark forward pass.

        Args:
            model: Model to benchmark

        Returns:
            Average forward time in seconds
        """
        model.eval()
        dummy_input = self._create_dummy_input()

        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup):
                _ = model(dummy_input)

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(self.num_runs):
                start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

                if start is not None:
                    start.record()
                else:
                    start = __import__('time').perf_counter()

                _ = model(dummy_input)

                if end is not None:
                    end.record()
                    torch.cuda.synchronize()
                    times.append(start.elapsed_time(end) / 1000)
                else:
                    end = __import__('time').perf_counter()
                    times.append(end - start)

        return sum(times) / len(times)

    def _benchmark_backward(self, model: torch.nn.Module) -> float:
        """Benchmark forward + backward pass.

        Args:
            model: Model to benchmark

        Returns:
            Average training time in seconds
        """
        model.train()
        dummy_input = self._create_dummy_input()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Warmup
        for _ in range(self.warmup):
            optimizer.zero_grad()
            output = model(dummy_input)
            if isinstance(output, (list, tuple)):
                output = output[0]
            # Create dummy target
            target = torch.randint(0, 2, (self.batch_size, 2, self.input_size, self.input_size))
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

        # Benchmark
        times = []
        for _ in range(self.num_runs // 2):
            start = __import__('time').perf_counter()

            optimizer.zero_grad()
            output = model(dummy_input)
            if isinstance(output, (list, tuple)):
                output = output[0]
            target = torch.randint(0, 2, (self.batch_size, 2, self.input_size, self.input_size))
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            end = __import__('time').perf_counter()
            times.append(end - start)

        return sum(times) / len(times)

    def _benchmark_memory(self, model: torch.nn.Module) -> float:
        """Benchmark memory usage.

        Args:
            model: Model to benchmark

        Returns:
            Memory usage in MB
        """
        if not torch.cuda.is_available():
            return 0.0

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        model = model.cuda()
        dummy_input = self._create_dummy_input()
        dummy_input = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in dummy_input.items()}

        with torch.no_grad():
            _ = model(dummy_input)

        memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        # Free
        model = model.cpu()
        torch.cuda.empty_cache()

        return memory_mb

    def run_feature(self) -> BenchmarkResult:
        """Run benchmark for feature adapter.

        Returns:
            Benchmark result
        """
        config = ExperimentConfig(
            name="benchmark_feature",
            task="flood",
            adaptation="feature",
            backbone="terramind_base",
        )

        model = ExperimentBuilder.build(config)

        if model is None:
            # Return placeholder if model not available
            return BenchmarkResult(
                adapter_type="feature",
                trainable_params=0,
                total_params=0,
                forward_time=0.0,
                backward_time=0.0,
                memory_mb=0.0,
            )

        total, trainable = self._count_params(model)
        forward_time = self._benchmark_forward(model)
        backward_time = self._benchmark_backward(model)
        memory_mb = self._benchmark_memory(model)

        result = BenchmarkResult(
            adapter_type="feature",
            trainable_params=trainable,
            total_params=total,
            forward_time=forward_time,
            backward_time=backward_time,
            memory_mb=memory_mb,
        )

        self.benchmark.add_result(result)
        return result

    def run_lora(self) -> BenchmarkResult:
        """Run benchmark for LoRA adapter.

        Returns:
            Benchmark result
        """
        config = ExperimentConfig(
            name="benchmark_lora",
            task="flood",
            adaptation="lora",
            backbone="terramind_base",
        )

        model = ExperimentBuilder.build(config)

        if model is None:
            return BenchmarkResult(
                adapter_type="lora",
                trainable_params=0,
                total_params=0,
                forward_time=0.0,
                backward_time=0.0,
                memory_mb=0.0,
            )

        total, trainable = self._count_params(model)
        forward_time = self._benchmark_forward(model)
        backward_time = self._benchmark_backward(model)
        memory_mb = self._benchmark_memory(model)

        result = BenchmarkResult(
            adapter_type="lora",
            trainable_params=trainable,
            total_params=total,
            forward_time=forward_time,
            backward_time=backward_time,
            memory_mb=memory_mb,
        )

        self.benchmark.add_result(result)
        return result

    def run_hybrid(self) -> BenchmarkResult:
        """Run benchmark for hybrid adapter.

        Returns:
            Benchmark result
        """
        config = ExperimentConfig(
            name="benchmark_hybrid",
            task="flood",
            adaptation="hybrid",
            backbone="terramind_base",
        )

        model = ExperimentBuilder.build(config)

        if model is None:
            return BenchmarkResult(
                adapter_type="hybrid",
                trainable_params=0,
                total_params=0,
                forward_time=0.0,
                backward_time=0.0,
                memory_mb=0.0,
            )

        total, trainable = self._count_params(model)
        forward_time = self._benchmark_forward(model)
        backward_time = self._benchmark_backward(model)
        memory_mb = self._benchmark_memory(model)

        result = BenchmarkResult(
            adapter_type="hybrid",
            trainable_params=trainable,
            total_params=total,
            forward_time=forward_time,
            backward_time=backward_time,
            memory_mb=memory_mb,
        )

        self.benchmark.add_result(result)
        return result

    def run_fullft(self) -> BenchmarkResult:
        """Run benchmark for full fine-tuning.

        Returns:
            Benchmark result
        """
        config = ExperimentConfig(
            name="benchmark_fullft",
            task="flood",
            adaptation="full_ft",
            backbone="terramind_base",
        )

        model = ExperimentBuilder.build(config)

        if model is None:
            return BenchmarkResult(
                adapter_type="full_ft",
                trainable_params=0,
                total_params=0,
                forward_time=0.0,
                backward_time=0.0,
                memory_mb=0.0,
            )

        total, trainable = self._count_params(model)
        forward_time = self._benchmark_forward(model)
        backward_time = self._benchmark_backward(model)
        memory_mb = self._benchmark_memory(model)

        result = BenchmarkResult(
            adapter_type="full_ft",
            trainable_params=trainable,
            total_params=total,
            forward_time=forward_time,
            backward_time=backward_time,
            memory_mb=memory_mb,
        )

        self.benchmark.add_result(result)
        return result

    def run_all(self) -> list[dict]:
        """Run benchmarks for all adapter types.

        Returns:
            List of result dictionaries
        """
        self.benchmark.clear()

        self.run_feature()
        self.run_lora()
        self.run_hybrid()
        self.run_fullft()

        return self.benchmark.summary()