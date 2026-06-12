"""tests/test_fit.py

Tests for fit module.
"""
from geofm.training.fit import fit


def test_fit_exists():
    """Test that fit function exists."""
    assert fit is not None


def test_fit_is_callable():
    """Test that fit is callable."""
    assert callable(fit)


def test_fit_signature():
    """Test fit function has expected parameters."""
    import inspect
    sig = inspect.signature(fit)
    params = list(sig.parameters.keys())

    assert "runner" in params
    assert "train_loader" in params
    assert "val_loader" in params
    assert "epochs" in params
    assert "logger" in params


def test_fit_returns_history():
    """Test that fit returns a history dictionary."""
    # Mock runner with empty loaders
    class MockRunner:
        def train_epoch(self, loader, task=None):
            return 0.0

        def validate_epoch(self, loader, task=None):
            return 0.0

    runner = MockRunner()
    history = fit(
        runner=runner,
        train_loader=[],
        val_loader=[],
        epochs=1,
    )

    assert isinstance(history, dict)
    assert "train_loss" in history
    assert "val_loss" in history