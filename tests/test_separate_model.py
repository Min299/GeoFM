"""tests/test_separate_model.py

Tests for SeparateGeoFM model.
"""
import torch.nn as nn

from geofm.models.multitask.separate_model import (
    SeparateGeoFM,
)


class Dummy(nn.Module):
    """Dummy model for testing."""

    def forward(
        self,
        x,
    ):
        return x


def test_init():
    """Test SeparateGeoFM initialization."""
    model = SeparateGeoFM(
        {
            "flood": Dummy(),
        }
    )

    assert (
        "flood"
        in model.available_tasks()
    )


def test_available_tasks():
    """Test available_tasks method."""
    model = SeparateGeoFM({
        "flood": Dummy(),
        "burn": Dummy(),
        "lulc": Dummy(),
    })

    tasks = model.available_tasks()
    assert "flood" in tasks
    assert "burn" in tasks
    assert "lulc" in tasks


def test_forward():
    """Test forward pass with valid task."""
    model = SeparateGeoFM({
        "flood": Dummy(),
    })

    result = model({"input": "data"}, task_name="flood")
    assert result is not None


def test_forward_unknown_task():
    """Test forward pass with unknown task raises KeyError."""
    model = SeparateGeoFM({
        "flood": Dummy(),
    })

    try:
        model({"input": "data"}, task_name="unknown_task")
        assert False, "Should have raised KeyError"
    except KeyError as e:
        assert "unknown_task" in str(e)