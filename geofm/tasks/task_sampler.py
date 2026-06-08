"""geofm.tasks.task_sampler

Task samplers for multi-task learning.

Provides various sampling strategies for selecting tasks during training.
"""
from __future__ import annotations

from typing import List, Dict, Optional, Callable
import random
from dataclasses import dataclass


@dataclass
class TaskSample:
    """A sampled task with metadata."""

    task_name: str
    weight: float = 1.0
    priority: float = 1.0


class TaskSampler:
    """Base task sampler.

    Randomly samples tasks from a list.

    Usage:
        sampler = TaskSampler(["flood", "burn", "lulc"])
        task = sampler.sample()  # Returns random task
    """

    def __init__(
        self,
        tasks: List[str],
        weights: Optional[List[float]] = None,
    ):
        """Initialize task sampler.

        Args:
            tasks: List of task names
            weights: Optional list of task weights
        """
        self.tasks = tasks

        if weights is None:
            self.weights = [1.0] * len(tasks)
        else:
            if len(weights) != len(tasks):
                raise ValueError(
                    f"Weights length ({len(weights)}) must match tasks length ({len(tasks)})"
                )
            self.weights = weights

    def sample(self) -> str:
        """Sample a random task.

        Returns:
            Sampled task name
        """
        return random.choice(self.tasks)

    def sample_with_weight(self) -> str:
        """Sample a task based on weights.

        Returns:
            Sampled task name
        """
        return random.choices(self.tasks, weights=self.weights, k=1)[0]

    def sample_batch(self, batch_size: int) -> List[str]:
        """Sample multiple tasks.

        Args:
            batch_size: Number of tasks to sample

        Returns:
            List of sampled task names
        """
        return random.choices(self.tasks, weights=self.weights, k=batch_size)

    def task_list(self) -> List[str]:
        """Get list of all tasks.

        Returns:
            List of task names
        """
        return self.tasks.copy()

    def set_weights(self, weights: List[float]) -> None:
        """Update task weights.

        Args:
            weights: New weights
        """
        if len(weights) != len(self.tasks):
            raise ValueError(
                f"Weights length ({len(weights)}) must match tasks length ({len(self.tasks)})"
            )
        self.weights = weights


class TaskSamplerWithRepetition(TaskSampler):
    """Sampler that allows task repetition within a batch."""

    def __init__(
        self,
        tasks: List[str],
        weights: Optional[List[float]] = None,
    ):
        """Initialize sampler with repetition.

        Args:
            tasks: List of task names
            weights: Optional list of task weights
        """
        super().__init__(tasks, weights)

    def sample_unique(self, num_samples: int) -> List[str]:
        """Sample unique tasks without repetition.

        Args:
            num_samples: Number of unique tasks to sample

        Returns:
            List of unique task names
        """
        if num_samples > len(self.tasks):
            raise ValueError(
                f"Cannot sample {num_samples} unique tasks from {len(self.tasks)} tasks"
            )
        return random.sample(self.tasks, k=num_samples)


class StratifiedSampler(TaskSampler):
    """Sampler that maintains stratification across batches.

    Ensures each batch contains a balanced representation of tasks.
    """

    def __init__(
        self,
        tasks: List[str],
        weights: Optional[List[float]] = None,
        min_samples_per_task: int = 1,
    ):
        """Initialize stratified sampler.

        Args:
            tasks: List of task names
            weights: Optional list of task weights
            min_samples_per_task: Minimum samples per task per batch
        """
        super().__init__(tasks, weights)
        self.min_samples_per_task = min_samples_per_task

    def sample_stratified(self, batch_size: int) -> List[str]:
        """Sample tasks with stratification.

        Ensures minimum samples per task, remaining slots filled proportionally.

        Args:
            batch_size: Total number of samples

        Returns:
            List of sampled task names
        """
        samples = []

        # Add minimum samples per task
        for _ in range(self.min_samples_per_task):
            for task in self.tasks:
                samples.append(task)

        # Fill remaining proportionally
        remaining = batch_size - len(samples)
        if remaining > 0:
            additional = self.sample_batch(remaining)
            samples.extend(additional)

        random.shuffle(samples)
        return samples[:batch_size]


class PrioritySampler(TaskSampler):
    """Sampler that respects task priorities.

    Higher priority tasks are sampled more frequently.
    """

    def __init__(
        self,
        tasks: List[str],
        priorities: List[float],
    ):
        """Initialize priority sampler.

        Args:
            tasks: List of task names
            priorities: List of task priorities (higher = more frequent)
        """
        super().__init__(tasks)
        self.priorities = priorities

    def sample(self) -> str:
        """Sample task based on priorities.

        Returns:
            Sampled task name
        """
        return random.choices(self.tasks, weights=self.priorities, k=1)[0]


class AdaptiveSampler(TaskSampler):
    """Sampler that adapts weights based on task performance.

    Tasks that are performing poorly get higher weights.
    """

    def __init__(
        self,
        tasks: List[str],
        initial_weights: Optional[List[float]] = None,
        adaptation_factor: float = 1.1,
    ):
        """Initialize adaptive sampler.

        Args:
            tasks: List of task names
            initial_weights: Optional initial weights
            adaptation_factor: Factor to adjust weights by
        """
        super().__init__(tasks, initial_weights)
        self.adaptation_factor = adaptation_factor
        self.performance_history: Dict[str, List[float]] = {
            task: [] for task in tasks
        }

    def update_performance(self, task: str, performance: float) -> None:
        """Update performance history for a task.

        Args:
            task: Task name
            performance: Performance metric (higher is better)
        """
        self.performance_history[task].append(performance)

    def sample(self) -> str:
        """Sample task based on recent performance.

        Tasks with lower recent performance get higher sampling weight.

        Returns:
            Sampled task name
        """
        # Calculate weights based on inverse of recent performance
        weights = []
        for task in self.tasks:
            history = self.performance_history[task]
            if not history:
                weights.append(1.0)
            else:
                # Use inverse of average recent performance
                avg_perf = sum(history[-5:]) / len(history[-5:])
                weights.append(1.0 / (avg_perf + 1e-6))

        return random.choices(self.tasks, weights=weights, k=1)[0]


class CurriculumSampler(TaskSampler):
    """Sampler that implements curriculum learning.

    Starts with easier tasks and gradually introduces harder tasks.
    """

    def __init__(
        self,
        tasks: List[str],
        difficulty_levels: List[int],
        current_level: int = 0,
    ):
        """Initialize curriculum sampler.

        Args:
            tasks: List of task names
            difficulty_levels: Difficulty level for each task
            current_level: Current difficulty level (0 = easiest)
        """
        super().__init__(tasks)
        self.difficulty_levels = difficulty_levels
        self.current_level = current_level

        # Sort by difficulty
        self.task_difficulties = list(zip(tasks, difficulty_levels))
        self.task_difficulties.sort(key=lambda x: x[1])

    def sample(self) -> str:
        """Sample task from current curriculum level.

        Returns:
            Sampled task name
        """
        # Filter tasks at or below current level
        available_tasks = [
            task for task, diff in self.task_difficulties
            if diff <= self.current_level
        ]

        if not available_tasks:
            # If no tasks at current level, use all tasks
            available_tasks = self.tasks

        return random.choice(available_tasks)

    def advance_level(self) -> None:
        """Advance to next difficulty level."""
        max_level = max(self.difficulty_levels)
        if self.current_level < max_level:
            self.current_level += 1

    def set_level(self, level: int) -> None:
        """Set curriculum level.

        Args:
            level: New difficulty level
        """
        max_level = max(self.difficulty_levels)
        self.current_level = min(level, max_level)