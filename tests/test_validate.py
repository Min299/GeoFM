"""tests/test_validate.py

Tests for validate module.
"""
from geofm.training.validate import validate


def test_validate_exists():
    """Test that validate function exists."""
    assert validate is not None


def test_validate_is_callable():
    """Test that validate is callable."""
    assert callable(validate)


def test_validate_signature():
    """Test validate function has expected parameters."""
    import inspect
    sig = inspect.signature(validate)
    params = list(sig.parameters.keys())

    assert "runner" in params
    assert "val_loader" in params


def test_validate_delegates_to_runner():
    """Test that validate delegates to runner.validate_epoch."""
    class MockRunner:
        def validate_epoch(self, loader, task=None):
            return 0.123

    runner = MockRunner()
    result = validate(runner=runner, val_loader=[])

    assert result == 0.123