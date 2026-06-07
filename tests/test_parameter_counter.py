"""tests/test_parameter_counter.py

Tests for parameter counting utilities.
"""
import pytest
import torch
import torch.nn as nn

from geofm.models.peft import (
    count_total_params,
    count_trainable_params,
    count_frozen_params,
    trainable_ratio,
    peft_percentage,
    format_params,
    verify_peft_ready,
    ParameterCounter,
)


class SimpleModel(nn.Module):
    """Simple test model."""

    def __init__(self, trainable_pct=0.5):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 10)

        # Freeze some params based on trainable_pct
        if trainable_pct < 1.0:
            for param in self.linear2.parameters():
                param.requires_grad = False


@pytest.fixture
def simple_model():
    """Create a simple test model."""
    return SimpleModel()


@pytest.fixture
def mixed_model():
    """Create a model with mixed trainable/frozen params."""
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.Linear(50, 10),
    )

    # Freeze first layer
    for param in model[0].parameters():
        param.requires_grad = False

    return model


class TestCountTotalParams:
    """Test total parameter counting."""

    def test_count_total(self, simple_model):
        """Should count all parameters."""
        total = count_total_params(simple_model)

        # linear1: 10*20 + 20 = 220
        # linear2: 20*10 + 10 = 210
        # Total: 430
        assert total == 430

    def test_count_with_nested(self):
        """Should handle nested modules."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Sequential(
                nn.Linear(20, 30),
                nn.Linear(30, 10),
            )
        )

        total = count_total_params(model)
        assert total > 0


class TestCountTrainableParams:
    """Test trainable parameter counting."""

    def test_count_trainable_all(self):
        """All params trainable."""
        model = nn.Linear(10, 20)
        count = count_trainable_params(model)

        # 10*20 + 20 = 220
        assert count == 220

    def test_count_trainable_partial(self, mixed_model):
        """Count only trainable params."""
        trainable = count_trainable_params(mixed_model)

        # First layer frozen (100*50 + 50 = 5050)
        # Second layer trainable (50*10 + 10 = 510)
        # Trainable should be 510
        assert trainable == 510


class TestCountFrozenParams:
    """Test frozen parameter counting."""

    def test_count_frozen(self, mixed_model):
        """Count frozen params."""
        frozen = count_frozen_params(mixed_model)

        # First layer frozen: 100*50 + 50 = 5050
        assert frozen == 5050


class TestTrainableRatio:
    """Test trainable ratio calculation."""

    def test_ratio_all_trainable(self):
        """100% trainable returns 1.0."""
        model = nn.Linear(10, 20)
        ratio = trainable_ratio(model)

        assert ratio == 1.0

    def test_ratio_none_trainable(self):
        """0% trainable returns 0.0."""
        model = nn.Linear(10, 20)
        for param in model.parameters():
            param.requires_grad = False

        ratio = trainable_ratio(model)

        assert ratio == 0.0

    def test_ratio_partial(self, mixed_model):
        """Partial trainable returns correct ratio."""
        total = count_total_params(mixed_model)
        trainable = count_trainable_params(mixed_model)

        expected_ratio = trainable / total
        actual_ratio = trainable_ratio(mixed_model)

        assert actual_ratio == expected_ratio


class TestPeftPercentage:
    """Test PEFT percentage calculation."""

    def test_peft_percentage(self):
        """Should return percentage (0-100)."""
        model = nn.Linear(100, 100)

        # All trainable
        assert peft_percentage(model) == 100.0

        # Freeze all
        for param in model.parameters():
            param.requires_grad = False

        assert peft_percentage(model) == 0.0


class TestFormatParams:
    """Test parameter formatting."""

    def test_format_thousands(self):
        """Format values in thousands."""
        assert format_params(500) == "500"
        assert format_params(1000) == "1.0K"
        assert format_params(5500) == "5.5K"

    def test_format_millions(self):
        """Format values in millions."""
        assert format_params(1_000_000) == "1.0M"
        assert format_params(85_000_000) == "85.0M"
        assert format_params(2_500_000) == "2.5M"

    def test_format_small(self):
        """Format small values."""
        assert format_params(100) == "100"
        assert format_params(0) == "0"


class TestVerifyPeftReady:
    """Test PEFT readiness verification."""

    def test_under_threshold(self):
        """Should pass when under threshold."""
        model = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.Linear(1000, 10),
        )

        # Freeze first layer
        for param in model[0].parameters():
            param.requires_grad = False

        # Only 10*1000 + 10 = 10,010 params trainable out of ~2M
        result = verify_peft_ready(model, max_ratio=0.05)

        assert result is True

    def test_over_threshold(self):
        """Should fail when over threshold."""
        model = nn.Linear(100, 100)
        # Most params trainable
        result = verify_peft_ready(model, max_ratio=0.05)

        assert result is False


class TestParameterCounter:
    """Test ParameterCounter utility class."""

    def test_initialization(self, simple_model):
        """Should initialize with model."""
        counter = ParameterCounter(simple_model)

        assert counter.model is simple_model
        assert counter.initial_total > 0

    def test_get_stats(self, simple_model):
        """Should return statistics dict."""
        counter = ParameterCounter(simple_model)

        stats = counter.get_stats()

        assert "total" in stats
        assert "trainable" in stats
        assert "frozen" in stats
        assert "ratio" in stats
        assert "peft_pct" in stats

    def test_print_stats(self, simple_model, capsys):
        """Should print formatted stats."""
        counter = ParameterCounter(simple_model)
        counter.print_stats()

        captured = capsys.readouterr()
        assert "Total:" in captured.out
        assert "Trainable:" in captured.out
        assert "Ratio:" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])