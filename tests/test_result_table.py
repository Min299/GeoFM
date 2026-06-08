"""tests/test_result_table.py

Tests for ResultTable.
"""
import pytest
from pathlib import Path

from geofm.logging.result_table import ResultTable


class TestResultTable:
    """Test ResultTable class."""

    def test_empty_table(self):
        """Empty table should have 0 rows."""
        table = ResultTable()
        assert len(table) == 0

    def test_add_result(self):
        """Adding result should increase row count."""
        table = ResultTable()

        table.add_result(
            experiment="exp01",
            task="flood",
            adapter="lora",
            metrics={"iou": 0.85, "dice": 0.92},
        )

        assert len(table) == 1
        assert table.rows[0]["experiment"] == "exp01"
        assert table.rows[0]["iou"] == 0.85

    def test_save_load_csv(self, tmp_path):
        """Save and load should preserve data."""
        table = ResultTable()

        table.add_result("exp1", "flood", "feature", {"iou": 0.9, "dice": 0.95})
        table.add_result("exp2", "flood", "lora", {"iou": 0.85, "dice": 0.90})

        csv_path = tmp_path / "results.csv"
        table.save_csv(csv_path)

        loaded = ResultTable.load_csv(csv_path)

        assert len(loaded.rows) == 2
        assert loaded.rows[0]["experiment"] == "exp1"
        assert loaded.rows[1]["adapter"] == "lora"

    def test_multiple_experiments(self):
        """Should handle multiple experiments."""
        table = ResultTable()

        for i in range(5):
            table.add_result(
                f"exp{i}",
                "flood",
                "lora",
                {"iou": 0.8 + i * 0.02},
            )

        assert len(table) == 5

    def test_different_tasks(self):
        """Should handle different tasks."""
        table = ResultTable()

        table.add_result("exp1", "flood", "lora", {"iou": 0.85})
        table.add_result("exp2", "burn", "lora", {"iou": 0.75})
        table.add_result("exp3", "lulc", "lora", {"accuracy": 0.80})

        assert table.rows[0]["task"] == "flood"
        assert table.rows[1]["task"] == "burn"
        assert table.rows[2]["task"] == "lulc"

    def test_pivot_table(self):
        """Should create pivot table correctly."""
        table = ResultTable()

        table.add_result("exp1", "flood", "lora", {"iou": 0.85})
        table.add_result("exp2", "flood", "feature", {"iou": 0.80})
        table.add_result("exp3", "burn", "lora", {"iou": 0.75})
        table.add_result("exp4", "burn", "feature", {"iou": 0.70})

        pivot = table.get_pivot_table(
            index="adapter",
            columns="task",
            values="iou",
        )

        assert pivot.loc["lora", "flood"] == 0.85
        assert pivot.loc["feature", "flood"] == 0.80
        assert pivot.loc["lora", "burn"] == 0.75

    def test_summary(self):
        """Should compute summary statistics."""
        table = ResultTable()

        # Add metrics with all required columns
        table.add_result("exp1", "flood", "lora", {"iou": 0.85, "dice": 0.92, "f1": 0.90})
        table.add_result("exp2", "flood", "lora", {"iou": 0.90, "dice": 0.95, "f1": 0.92})
        table.add_result("exp3", "flood", "feature", {"iou": 0.80, "dice": 0.85, "f1": 0.82})

        summary = table.get_summary(group_by="adapter")

        # LoRA should have mean (0.85 + 0.90) / 2 = 0.875
        lora_mean = summary.loc["lora", ("iou", "mean")]
        assert abs(lora_mean - 0.875) < 0.01, f"LoRA mean={lora_mean}"


class TestResultTableEdgeCases:
    """Test edge cases."""

    def test_empty_save(self, tmp_path):
        """Saving empty table should create file with headers."""
        table = ResultTable()
        csv_path = tmp_path / "empty.csv"

        table.save_csv(csv_path)

        # Should create file
        assert csv_path.exists()

        # Load and verify
        loaded = ResultTable.load_csv(csv_path)
        assert len(loaded.rows) == 0

    def test_load_nonexistent(self, tmp_path):
        """Loading nonexistent file should raise error."""
        table = ResultTable()
        csv_path = tmp_path / "nonexistent.csv"

        with pytest.raises(FileNotFoundError):
            ResultTable.load_csv(csv_path)

    def test_repr(self):
        """repr should show row count."""
        table = ResultTable()
        table.add_result("exp1", "flood", "lora", {"iou": 0.85})

        assert "1" in repr(table)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])