"""tests/test_independent_model_builder.py

Tests for IndependentModelBuilder.
"""
import torch.nn as nn

from geofm.builders.independent_model_builder import (
    IndependentModelBuilder,
)


class Dummy(nn.Module):
    """Dummy model for testing."""

    def forward(
        self,
        x,
    ):
        return x


def test_builder():
    """Test basic builder functionality."""
    model = (
        IndependentModelBuilder()
        .add_task_model(
            "flood",
            Dummy(),
        )
        .build()
    )

    assert model is not None


def test_builder_multiple_tasks():
    """Test builder with multiple tasks."""
    model = (
        IndependentModelBuilder()
        .add_task_model("flood", Dummy())
        .add_task_model("burn", Dummy())
        .add_task_model("lulc", Dummy())
        .build()
    )

    tasks = model.available_tasks()
    assert "flood" in tasks
    assert "burn" in tasks
    assert "lulc" in tasks


def test_builder_empty_raises():
    """Test that building without tasks raises ValueError."""
    builder = IndependentModelBuilder()

    try:
        builder.build()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "No task models registered" in str(e)