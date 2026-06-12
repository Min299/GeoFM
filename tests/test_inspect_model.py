"""tests/test_inspect_model.py

Smoke tests for inspect_model.py script.
"""
import pytest


def test_inspect_script_import():
    """Inspect script should be importable."""
    import scripts.inspect_model

    assert hasattr(scripts.inspect_model, "main")
    assert hasattr(scripts.inspect_model, "count_parameters")
    assert hasattr(scripts.inspect_model, "inspect")


def test_count_parameters():
    """count_parameters should return correct counts."""
    import torch
    from scripts.inspect_model import count_parameters

    model = torch.nn.Linear(10, 5)

    total, trainable, frozen = count_parameters(model)

    assert total == 10 * 5 + 5  # weights + bias
    assert trainable == total  # Linear is trainable by default
    assert frozen == 0


def test_count_parameters_with_frozen():
    """count_parameters should handle frozen layers."""
    import torch
    from scripts.inspect_model import count_parameters

    model = torch.nn.Linear(10, 5)

    # Freeze the layer
    for p in model.parameters():
        p.requires_grad = False

    total, trainable, frozen = count_parameters(model)

    # 10*5 weights + 5 bias = 55 total
    assert total == 55
    assert trainable == 0
    assert frozen == 55


def test_inspect_function():
    """inspect should not raise errors."""
    import torch
    from scripts.inspect_model import inspect

    model = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 2),
    )

    # Should not raise
    inspect(model, "TestModel")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])