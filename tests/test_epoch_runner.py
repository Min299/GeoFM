"""tests/test_epoch_runner.py

Tests for epoch_runner module.
"""
import torch
import torch.nn as nn
from geofm.training.epoch_runner import EpochRunner


def test_runner_init():
    """Test EpochRunner initialization."""
    runner = EpochRunner(
        model=None,
        optimizer=None,
        criterion=None,
    )
    assert runner is not None
    assert runner.model is None
    assert runner.optimizer is None
    assert runner.criterion is None


def test_runner_has_train_epoch():
    """Test that runner has train_epoch method."""
    runner = EpochRunner(
        model=None,
        optimizer=None,
        criterion=None,
    )
    assert hasattr(runner, "train_epoch")
    assert callable(runner.train_epoch)


def test_runner_has_validate_epoch():
    """Test that runner has validate_epoch method."""
    runner = EpochRunner(
        model=None,
        optimizer=None,
        criterion=None,
    )
    assert hasattr(runner, "validate_epoch")
    assert callable(runner.validate_epoch)


def test_runner_with_mock_components():
    """Test EpochRunner with mock components."""
    mock_model = nn.Linear(10, 5)
    mock_optimizer = torch.optim.SGD(mock_model.parameters(), lr=0.01)
    mock_criterion = nn.CrossEntropyLoss()

    runner = EpochRunner(
        model=mock_model,
        optimizer=mock_optimizer,
        criterion=mock_criterion,
    )

    assert runner.model is mock_model
    assert runner.optimizer is mock_optimizer
    assert runner.criterion is mock_criterion


def test_runner_train_epoch_empty_loader():
    """Test train_epoch with empty loader returns 0.0."""
    runner = EpochRunner(
        model=None,
        optimizer=None,
        criterion=None,
    )

    # Empty loader
    empty_loader = []

    result = runner.train_epoch(empty_loader)
    assert result == 0.0


def test_runner_validate_epoch_empty_loader():
    """Test validate_epoch with empty loader returns 0.0."""
    runner = EpochRunner(
        model=None,
        optimizer=None,
        criterion=None,
    )

    # Empty loader
    empty_loader = []

    result = runner.validate_epoch(empty_loader)
    assert result == 0.0