"""tests/test_runtime.py

Tests for Runtime.
"""
from geofm.integration.runtime import (
    Runtime,
)


def test_runtime_exists():
    """Test that Runtime class exists."""
    assert Runtime is not None


def test_runtime_init():
    """Test Runtime initialization with config."""
    class MockConfig:
        class Training:
            epochs = 10
        training = Training()

    runtime = Runtime(cfg=MockConfig())

    assert runtime.cfg is not None
    assert runtime.cfg.training.epochs == 10