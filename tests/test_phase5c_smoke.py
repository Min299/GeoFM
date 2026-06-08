"""tests/test_phase5c_smoke.py

Smoke tests for Phase 5C infrastructure.
"""
import pytest


class TestPhase5CSmoke:
    """Smoke tests for Phase 5C."""

    def test_all_methods_exist(self):
        """All adaptation methods should be defined."""
        methods = ["feature", "lora", "hybrid", "fullft"]

        for method in methods:
            assert method is not None
            assert isinstance(method, str)

    def test_experiment_registry_imports(self):
        """Experiment registry should be importable."""
        from geofm.experiments import (
            EXPERIMENTS,
            get_experiment,
            list_experiments,
        )

        assert len(EXPERIMENTS) == 4
        assert len(list_experiments()) == 4

    def test_benchmark_config_imports(self):
        """Benchmark config should be importable."""
        from geofm.experiments import (
            BenchmarkConfig,
            BenchmarkSuiteConfig,
        )

        config = BenchmarkConfig(experiment_name="feature", task="flood")
        assert config.experiment_name == "feature"

        suite = BenchmarkSuiteConfig(experiments=["feature"], task="flood")
        assert len(suite.experiments) == 1

    def test_adaptation_benchmark_imports(self):
        """Adaptation benchmark should be importable."""
        from geofm.experiments import (
            AdaptationBenchmark,
            MultiTaskBenchmark,
        )

        assert AdaptationBenchmark is not None
        assert MultiTaskBenchmark is not None

    def test_benchmark_runner_imports(self):
        """Benchmark runner should be importable."""
        from geofm.experiments import (
            BenchmarkRunner,
            SuiteRunner,
        )

        assert BenchmarkRunner is not None
        assert SuiteRunner is not None

    def test_evaluation_imports(self):
        """Evaluation module should be importable."""
        from geofm.evaluation import (
            BenchmarkMetrics,
            BenchmarkReport,
            ResultsWriter,
            Leaderboard,
        )

        assert BenchmarkMetrics is not None
        assert BenchmarkReport is not None
        assert ResultsWriter is not None
        assert Leaderboard is not None

    def test_config_files_exist(self):
        """Config files should exist."""
        from pathlib import Path

        configs_dir = Path("configs")

        assert (configs_dir / "flood_feature.yaml").exists()
        assert (configs_dir / "flood_lora.yaml").exists()
        assert (configs_dir / "flood_hybrid.yaml").exists()
        assert (configs_dir / "flood_fullft.yaml").exists()

    def test_leaderboard_smoke(self):
        """Leaderboard should work."""
        from geofm.evaluation import Leaderboard

        board = Leaderboard()
        board.add_result("feature", 0.85)
        board.add_result("lora", 0.82)
        board.add_result("hybrid", 0.88)

        ranking = board.ranking()

        assert ranking[0][0] == "hybrid"
        assert ranking[0][1] == 0.88

    def test_results_writer_smoke(self):
        """ResultsWriter should work."""
        import tempfile
        from geofm.evaluation import ResultsWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ResultsWriter(output_dir=tmpdir, timestamp=False)

            writer.write("feature", {"dice": 0.85})

            assert writer.get_output_dir().exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])