"""geofm.training.multitask_trainer

Multi-task trainer for GeoFM.

Provides training loop for multi-task learning with task scheduling.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import torch
import torch.nn as nn

from geofm.tasks.task_scheduler import TaskScheduler
from geofm.tasks.task_batcher import TaskBatcher
from geofm.training.multitask_step import multitask_step


@dataclass
class TrainingMetrics:
    """Metrics from a training step or epoch."""

    loss: float = 0.0
    task: str = ""
    grad_norm: float = 0.0
    step: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "loss": self.loss,
            "task": self.task,
            "grad_norm": self.grad_norm,
            "step": self.step,
        }


@dataclass
class EpochMetrics:
    """Metrics from a full training epoch."""

    avg_loss: float = 0.0
    total_steps: int = 0
    task_losses: Dict[str, List[float]] = field(default_factory=dict)
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "avg_loss": self.avg_loss,
            "total_steps": self.total_steps,
            "task_losses": self.task_losses,
            "duration_seconds": self.duration_seconds,
        }


class MultiTaskTrainer:
    """Trainer for multi-task learning.

    Supports various task scheduling strategies and training modes.

    Usage:
        trainer = MultiTaskTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            tasks=["flood", "burn", "lulc"],
        )
        epoch_metrics = trainer.train_epoch(task_batcher, num_steps=100)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        tasks: List[str],
        device: Optional[torch.device] = None,
        scheduler: Optional[TaskScheduler] = None,
        clip_grad: Optional[float] = None,
    ):
        """Initialize multi-task trainer.

        Args:
            model: The model to train
            optimizer: Optimizer
            criterion: Loss function
            tasks: List of task names
            device: Optional device to run on
            scheduler: Optional custom task scheduler
            clip_grad: Optional gradient clipping value
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.tasks = tasks
        self.device = device
        self.clip_grad = clip_grad

        # Create default scheduler if not provided
        if scheduler is None:
            self.scheduler = TaskScheduler(tasks)
        else:
            self.scheduler = scheduler

        self.step_count = 0

    def train_epoch(
        self,
        task_batcher: TaskBatcher,
        num_steps: int,
    ) -> EpochMetrics:
        """Train for one epoch.

        Args:
            task_batcher: TaskBatcher for getting batches
            num_steps: Number of training steps

        Returns:
            EpochMetrics with training statistics
        """
        losses = []
        task_losses: Dict[str, List[float]] = {task: [] for task in self.tasks}

        for _ in range(num_steps):
            # Get next task from scheduler
            task = self.scheduler.next_task()

            # Get batch for this task
            batch = task_batcher.get_batch(task)

            # Execute training step
            result = multitask_step(
                self.model,
                batch.to_dict(),
                task,
                self.criterion,
                self.optimizer,
                self.device,
            )

            loss = result["loss"]
            losses.append(loss)
            task_losses[task].append(loss)
            self.step_count += 1

        avg_loss = sum(losses) / len(losses) if losses else 0.0

        return EpochMetrics(
            avg_loss=avg_loss,
            total_steps=num_steps,
            task_losses=task_losses,
        )

    def train_epoch_with_grad_clip(
        self,
        task_batcher: TaskBatcher,
        num_steps: int,
    ) -> EpochMetrics:
        """Train for one epoch with gradient clipping.

        Args:
            task_batcher: TaskBatcher for getting batches
            num_steps: Number of training steps

        Returns:
            EpochMetrics with training statistics
        """
        from geofm.training.multitask_step import multitask_step_with_grad

        losses = []
        task_losses: Dict[str, List[float]] = {task: [] for task in self.tasks}

        for _ in range(num_steps):
            task = self.scheduler.next_task()
            batch = task_batcher.get_batch(task)

            result = multitask_step_with_grad(
                self.model,
                batch.to_dict(),
                task,
                self.criterion,
                self.optimizer,
                self.device,
                self.clip_grad,
            )

            loss = result["loss"]
            losses.append(loss)
            task_losses[task].append(loss)
            self.step_count += 1

        avg_loss = sum(losses) / len(losses) if losses else 0.0

        return EpochMetrics(
            avg_loss=avg_loss,
            total_steps=num_steps,
            task_losses=task_losses,
        )

    def evaluate(
        self,
        task_batcher: TaskBatcher,
        num_steps: int,
    ) -> EpochMetrics:
        """Evaluate the model.

        Args:
            task_batcher: TaskBatcher for getting batches
            num_steps: Number of evaluation steps

        Returns:
            EpochMetrics with evaluation statistics
        """
        self.model.eval()
        losses = []
        task_losses: Dict[str, List[float]] = {task: [] for task in self.tasks}

        with torch.no_grad():
            for _ in range(num_steps):
                task = self.scheduler.next_task()
                batch = task_batcher.get_batch(task)

                inputs = batch.inputs
                targets = batch.targets

                if self.device:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                output = self.model(inputs, task=task)
                loss = self.criterion(output, targets)

                losses.append(loss.item())
                task_losses[task].append(loss.item())

        self.model.train()

        avg_loss = sum(losses) / len(losses) if losses else 0.0

        return EpochMetrics(
            avg_loss=avg_loss,
            total_steps=num_steps,
            task_losses=task_losses,
        )

    def get_step_count(self) -> int:
        """Get total number of training steps.

        Returns:
            Step count
        """
        return self.step_count

    def reset_step_count(self) -> None:
        """Reset step counter."""
        self.step_count = 0


class GradientMultiTaskTrainer(MultiTaskTrainer):
    """Multi-task trainer with gradient accumulation support."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        tasks: List[str],
        accumulation_steps: int = 1,
        device: Optional[torch.device] = None,
    ):
        """Initialize gradient accumulation trainer.

        Args:
            model: The model to train
            optimizer: Optimizer
            criterion: Loss function
            tasks: List of task names
            accumulation_steps: Number of steps to accumulate gradients
            device: Optional device to run on
        """
        super().__init__(model, optimizer, criterion, tasks, device)
        self.accumulation_steps = accumulation_steps
        self.accumulation_count = 0

    def train_epoch(
        self,
        task_batcher: TaskBatcher,
        num_steps: int,
    ) -> EpochMetrics:
        """Train with gradient accumulation.

        Args:
            task_batcher: TaskBatcher for getting batches
            num_steps: Number of training steps (effective steps = num_steps / accumulation_steps)

        Returns:
            EpochMetrics with training statistics
        """
        losses = []
        task_losses: Dict[str, List[float]] = {task: [] for task in self.tasks}

        for _ in range(num_steps):
            task = self.scheduler.next_task()
            batch = task_batcher.get_batch(task)

            inputs = batch.inputs
            targets = batch.targets

            if self.device:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

            output = self.model(inputs, task=task)
            loss = self.criterion(output, targets)
            loss = loss / self.accumulation_steps

            loss.backward()

            self.accumulation_count += 1

            if self.accumulation_count >= self.accumulation_steps:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.accumulation_count = 0

            losses.append(loss.item() * self.accumulation_steps)
            task_losses[task].append(loss.item() * self.accumulation_steps)
            self.step_count += 1

        # Handle remaining accumulated gradients
        if self.accumulation_count > 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        avg_loss = sum(losses) / len(losses) if losses else 0.0

        return EpochMetrics(
            avg_loss=avg_loss,
            total_steps=num_steps,
            task_losses=task_losses,
        )


class AdaptiveMultiTaskTrainer(MultiTaskTrainer):
    """Multi-task trainer with dynamic task weighting."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        tasks: List[str],
        initial_weights: Optional[Dict[str, float]] = None,
        adaptation_rate: float = 0.01,
        device: Optional[torch.device] = None,
    ):
        """Initialize adaptive multi-task trainer.

        Args:
            model: The model to train
            optimizer: Optimizer
            criterion: Loss function
            tasks: List of task names
            initial_weights: Initial task weights
            adaptation_rate: Rate at which weights adapt
            device: Optional device to run on
        """
        super().__init__(model, optimizer, criterion, tasks, device)

        if initial_weights is None:
            self.task_weights = {task: 1.0 for task in tasks}
        else:
            self.task_weights = initial_weights

        self.adaptation_rate = adaptation_rate
        self.task_losses_history: Dict[str, List[float]] = {
            task: [] for task in tasks
        }

    def train_epoch(
        self,
        task_batcher: TaskBatcher,
        num_steps: int,
    ) -> EpochMetrics:
        """Train with adaptive task weighting.

        Args:
            task_batcher: TaskBatcher for getting batches
            num_steps: Number of training steps

        Returns:
            EpochMetrics with training statistics
        """
        losses = []
        task_losses: Dict[str, List[float]] = {task: [] for task in self.tasks}

        for _ in range(num_steps):
            task = self.scheduler.next_task()
            batch = task_batcher.get_batch(task)

            inputs = batch.inputs
            targets = batch.targets

            if self.device:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

            output = self.model(inputs, task=task)
            loss = self.criterion(output, targets)

            # Apply task weight
            weighted_loss = loss * self.task_weights[task]
            weighted_loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            # Record loss
            losses.append(loss.item())
            task_losses[task].append(loss.item())
            self.task_losses_history[task].append(loss.item())
            self.step_count += 1

            # Adapt task weights based on loss magnitude
            self._adapt_weights(task, loss.item())

        avg_loss = sum(losses) / len(losses) if losses else 0.0

        return EpochMetrics(
            avg_loss=avg_loss,
            total_steps=num_steps,
            task_losses=task_losses,
        )

    def _adapt_weights(self, task: str, loss: float) -> None:
        """Adapt task weight based on recent loss.

        Args:
            task: Task name
            loss: Recent loss value
        """
        history = self.task_losses_history[task]
        if len(history) < 2:
            return

        # Compare to previous loss
        prev_loss = history[-2]
        if loss > prev_loss:
            # Increase weight for underperforming tasks
            self.task_weights[task] *= (1 + self.adaptation_rate)
        else:
            # Slightly decrease weight for well-performing tasks
            self.task_weights[task] *= (1 - self.adaptation_rate * 0.1)

        # Clamp weights to reasonable range
        self.task_weights[task] = max(0.1, min(10.0, self.task_weights[task]))

    def get_task_weights(self) -> Dict[str, float]:
        """Get current task weights.

        Returns:
            Dictionary mapping task to weight
        """
        return self.task_weights.copy()