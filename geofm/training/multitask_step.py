"""geofm.training.multitask_step

Core multi-task training step.

Provides the fundamental training operation for multi-task learning.
"""
from __future__ import annotations

from typing import Dict, Any, Optional, Callable
import torch
import torch.nn as nn


def multitask_step(
    model: nn.Module,
    batch: Dict[str, Any],
    task: str,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Execute a single multi-task training step.

    Args:
        model: The model to train
        batch: Batch dictionary containing 'inputs' and 'targets'
        task: Current task name (e.g., 'flood', 'burn', 'lulc')
        criterion: Loss function
        optimizer: Optimizer
        device: Optional device to run on

    Returns:
        Dictionary with 'loss' and other metrics

    Usage:
        loss_dict = multitask_step(
            model=model,
            batch={"inputs": x, "targets": y},
            task="flood",
            criterion=criterion,
            optimizer=optimizer,
        )
    """
    # Move data to device if specified
    if device is not None:
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device)
    else:
        inputs = batch["inputs"]
        targets = batch["targets"]

    # Zero gradients
    optimizer.zero_grad()

    # Forward pass with task specification
    output = model(inputs, task=task)

    # Compute loss
    loss = criterion(output, targets)

    # Backward pass
    loss.backward()

    # Optimizer step
    optimizer.step()

    return {
        "loss": loss.item(),
        "task": task,
    }


def multitask_step_with_grad(
    model: nn.Module,
    batch: Dict[str, Any],
    task: str,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: Optional[torch.device] = None,
    clip_grad: Optional[float] = None,
) -> Dict[str, float]:
    """Execute a multi-task step with gradient clipping.

    Args:
        model: The model to train
        batch: Batch dictionary containing 'inputs' and 'targets'
        task: Current task name
        criterion: Loss function
        optimizer: Optimizer
        device: Optional device to run on
        clip_grad: Optional gradient clipping value

    Returns:
        Dictionary with 'loss', 'grad_norm', and other metrics
    """
    if device is not None:
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device)
    else:
        inputs = batch["inputs"]
        targets = batch["targets"]

    optimizer.zero_grad()

    output = model(inputs, task=task)
    loss = criterion(output, targets)

    loss.backward()

    # Clip gradients if specified
    grad_norm = 0.0
    if clip_grad is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            clip_grad,
        ).item()

    optimizer.step()

    return {
        "loss": loss.item(),
        "grad_norm": grad_norm,
        "task": task,
    }


def multitask_step_mixed_precision(
    model: nn.Module,
    batch: Dict[str, Any],
    task: str,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Execute a multi-task step with mixed precision.

    Args:
        model: The model to train
        batch: Batch dictionary containing 'inputs' and 'targets'
        task: Current task name
        criterion: Loss function
        optimizer: Optimizer
        scaler: Optional gradient scaler for mixed precision
        device: Optional device to run on

    Returns:
        Dictionary with 'loss' and other metrics
    """
    if device is not None:
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device)
    else:
        inputs = batch["inputs"]
        targets = batch["targets"]

    optimizer.zero_grad()

    # Forward with autocast for mixed precision
    if scaler is not None:
        with torch.cuda.amp.autocast():
            output = model(inputs, task=task)
            loss = criterion(output, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        output = model(inputs, task=task)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

    return {
        "loss": loss.item(),
        "task": task,
    }


class MultiTaskStepper:
    """Configurable multi-task step executor.

    Provides a class-based interface for multi-task training steps.

    Usage:
        stepper = MultiTaskStepper(model, optimizer, criterion)
        result = stepper.step(batch, task="flood")
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: Optional[torch.device] = None,
        clip_grad: Optional[float] = None,
        use_mixed_precision: bool = False,
    ):
        """Initialize multi-task stepper.

        Args:
            model: The model to train
            optimizer: Optimizer
            criterion: Loss function
            device: Optional device to run on
            clip_grad: Optional gradient clipping value
            use_mixed_precision: Whether to use mixed precision
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.clip_grad = clip_grad
        self.use_mixed_precision = use_mixed_precision
        self.scaler = None

        if use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()

    def step(
        self,
        batch: Dict[str, Any],
        task: str,
    ) -> Dict[str, float]:
        """Execute a training step.

        Args:
            batch: Batch dictionary
            task: Task name

        Returns:
            Dictionary with loss and metrics
        """
        if self.use_mixed_precision:
            return multitask_step_mixed_precision(
                self.model,
                batch,
                task,
                self.criterion,
                self.optimizer,
                self.scaler,
                self.device,
            )
        elif self.clip_grad is not None:
            return multitask_step_with_grad(
                self.model,
                batch,
                task,
                self.criterion,
                self.optimizer,
                self.device,
                self.clip_grad,
            )
        else:
            return multitask_step(
                self.model,
                batch,
                task,
                self.criterion,
                self.optimizer,
                self.device,
            )

    def state_dict(self) -> Dict:
        """Get state dict for checkpointing.

        Returns:
            State dictionary
        """
        state = {
            "optimizer": self.optimizer.state_dict(),
        }
        if self.scaler is not None:
            state["scaler"] = self.scaler.state_dict()
        return state

    def load_state_dict(self, state: Dict) -> None:
        """Load state dict from checkpoint.

        Args:
            state: State dictionary
        """
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scaler is not None and "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])