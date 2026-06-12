"""tests/test_peft_factory.py

Tests for PEFTFactory.
"""
from geofm.models.peft.peft_factory import (
    PEFTFactory,
)


class DummyModel:
    """Dummy model for testing."""

    def __call__(self, x):
        return x


def test_factory_lora():
    """Test PEFTFactory with lora strategy."""
    model = DummyModel()

    wrapped = PEFTFactory.build(
        "lora",
        model,
    )

    assert wrapped is not None
    assert wrapped.model is model


def test_factory_adapter():
    """Test PEFTFactory with adapter strategy."""
    model = DummyModel()

    wrapped = PEFTFactory.build(
        "adapter",
        model,
    )

    assert wrapped is not None
    assert wrapped.model is model


def test_factory_passthrough():
    """Test PEFTFactory returns original model for unknown strategy."""
    model = DummyModel()

    result = PEFTFactory.build(
        "unknown",
        model,
    )

    assert result is model