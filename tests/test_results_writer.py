"""tests/test_results_writer.py

Tests for results writer.
"""
import pytest
import tempfile
import os
from pathlib import Path


class TestResultsWriter:
    """Test results writer."""

    def test_init(self):
        """Writer should initialize."""
        from geofm.evaluation.results_writer import ResultsWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ResultsWriter(output_dir=tmpdir, timestamp=False)

            assert writer.output_dir == Path(tmpdir)
            assert writer.output_dir.exists()

    def test_write(self):
        """write should create file."""
        from geofm.evaluation.results_writer import ResultsWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ResultsWriter(output_dir=tmpdir, timestamp=False)

            metrics = {
                "dice": 0.85,
                "iou": 0.75,
                "trainable_params": 1000,
            }

            filepath = writer.write("feature", metrics)

            assert filepath.exists()
            assert filepath.name == "summary.json"

    def test_write_with_tag(self):
        """write with tag should create tagged file."""
        from geofm.evaluation.results_writer import ResultsWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ResultsWriter(output_dir=tmpdir, timestamp=False)

            metrics = {"dice": 0.85}

            filepath = writer.write("feature", metrics, tag="epoch_10")

            assert filepath.exists()
            assert filepath.name == "epoch_10.json"

    def test_write_history(self):
        """write_history should create history file."""
        from geofm.evaluation.results_writer import ResultsWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ResultsWriter(output_dir=tmpdir, timestamp=False)

            history = [
                {"epoch": 0, "loss": 0.9},
                {"epoch": 1, "loss": 0.7},
            ]

            filepath = writer.write_history("feature", history)

            assert filepath.exists()
            assert filepath.name == "history.json"

    def test_write_summary(self):
        """write_summary should create summary file."""
        from geofm.evaluation.results_writer import ResultsWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ResultsWriter(output_dir=tmpdir, timestamp=False)

            all_results = {
                "feature": {"dice": 0.85},
                "lora": {"dice": 0.82},
            }

            filepath = writer.write_summary(all_results)

            assert filepath.exists()
            assert filepath.name == "summary.json"

    def test_get_output_dir(self):
        """get_output_dir should return path."""
        from geofm.evaluation.results_writer import ResultsWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ResultsWriter(output_dir=tmpdir, timestamp=False)

            assert writer.get_output_dir() == Path(tmpdir)


class TestCSVResultsWriter:
    """Test CSV results writer."""

    def test_init(self):
        """CSV writer should initialize."""
        from geofm.evaluation.results_writer import CSVResultsWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CSVResultsWriter(output_dir=tmpdir)

            assert writer.output_dir == Path(tmpdir)

    def test_write_row(self):
        """write_row should create CSV file."""
        from geofm.evaluation.results_writer import CSVResultsWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = CSVResultsWriter(output_dir=tmpdir)

            metrics = {
                "dice": 0.85,
                "iou": 0.75,
            }

            filepath = writer.write_row("feature", metrics)

            assert filepath.exists()
            assert filepath.suffix == ".csv"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])