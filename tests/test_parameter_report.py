"""tests/test_parameter_report.py

Tests for parameter report utilities.
"""
import pytest
import torch.nn as nn


class TestParameterReport:
    """Test parameter counting utilities."""

    def test_count_parameters(self):
        """Should count all parameters."""
        from geofm.debug.parameter_report import count_parameters

        model = nn.Linear(10, 5)

        count = count_parameters(model)

        # Linear(10, 5) has 10*5=50 weights + 5 bias = 55 params
        assert count == 55

    def test_count_trainable_parameters(self):
        """Should count only trainable parameters."""
        from geofm.debug.parameter_report import count_trainable_parameters

        model = nn.Linear(10, 5)

        count = count_trainable_parameters(model)

        # All params are trainable by default
        assert count == 55

    def test_count_trainable_after_freeze(self):
        """Should count 0 after freezing."""
        from geofm.debug.parameter_report import count_trainable_parameters

        model = nn.Linear(10, 5)

        for p in model.parameters():
            p.requires_grad = False

        count = count_trainable_parameters(model)

        assert count == 0

    def test_parameter_report(self):
        """parameter_report should return dict with total, trainable, frozen."""
        from geofm.debug.parameter_report import parameter_report

        model = nn.Linear(10, 5)

        report = parameter_report(model)

        assert "total" in report
        assert "trainable" in report
        assert "frozen" in report

        assert report["total"] == 55
        assert report["trainable"] == 55
        assert report["frozen"] == 0

    def test_parameter_report_after_freeze(self):
        """parameter_report should reflect frozen state."""
        from geofm.debug.parameter_report import parameter_report

        model = nn.Linear(10, 5)

        for p in model.parameters():
            p.requires_grad = False

        report = parameter_report(model)

        assert report["total"] == 55
        assert report["trainable"] == 0
        assert report["frozen"] == 55

    def test_peft_ratio(self):
        """peft_ratio should return percentage."""
        from geofm.debug.parameter_report import peft_ratio

        model = nn.Linear(10, 5)

        ratio = peft_ratio(model)

        assert ratio == 100.0

    def test_peft_ratio_after_freeze(self):
        """peft_ratio should return 0 after freeze."""
        from geofm.debug.parameter_report import peft_ratio

        model = nn.Linear(10, 5)

        for p in model.parameters():
            p.requires_grad = False

        ratio = peft_ratio(model)

        assert ratio == 0.0

    def test_format_params(self):
        """format_params should format numbers nicely."""
        from geofm.debug.parameter_report import format_params

        assert format_params(100) == "100"
        assert format_params(1000) == "1.0K"
        assert format_params(1000000) == "1.0M"
        assert format_params(1500000) == "1.5M"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])