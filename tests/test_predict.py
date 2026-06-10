"""tests/test_predict.py

Tests for predict module.
"""
import torch
import torch.nn as nn
from geofm.training.predict import predict


def test_predict_exists():
    """Test that predict function exists."""
    assert predict is not None


def test_predict_is_callable():
    """Test that predict is callable."""
    assert callable(predict)


def test_predict_signature():
    """Test predict function has expected parameters."""
    import inspect
    sig = inspect.signature(predict)
    params = list(sig.parameters.keys())

    assert "model" in params
    assert "batch" in params
    assert "task" in params


def test_predict_with_mock_model():
    """Test predict with a simple mock model."""
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)

        def forward(self, x, task_name=None):
            return self.linear(x)

    model = MockModel()
    batch = torch.randn(2, 10)
    result = predict(model, batch)

    assert result is not None
    assert isinstance(result, torch.Tensor)
    assert result.shape == (2, 5)