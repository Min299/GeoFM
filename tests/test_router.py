"""tests/test_router.py

Tests for TaskRouter.
"""
from geofm.models.multitask.router import (
    TaskRouter,
)


class Dummy:
    """Dummy model for testing."""

    def __call__(
        self,
        batch,
        task_name=None,
    ):
        return batch


def test_router():
    """Test basic TaskRouter functionality."""
    router = TaskRouter(
        Dummy()
    )

    out = router(
        123,
        "flood",
    )

    assert out == 123


def test_router_call():
    """Test that __call__ equals forward."""
    router = TaskRouter(Dummy())

    result1 = router(42, "flood")
    result2 = router.forward(42, "flood")

    assert result1 == result2