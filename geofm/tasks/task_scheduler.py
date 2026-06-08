"""geofm.tasks.task_scheduler

Task scheduler for multi-task learning.

Provides round-robin and weighted scheduling of tasks.
"""
from __future__ import annotations

from typing import List, Dict, Optional
from dataclasses import dataclass
import random


@dataclass
class TaskConfig:
    """Configuration for a single task."""

    name: str
    weight: float = 1.0
    enabled: bool = True

    def __post_init__(self):
        """Validate task config."""
        if self.weight <= 0:
            raise ValueError(f"Weight must be positive, got {self.weight}")


class TaskScheduler:
    """Scheduler for multi-task learning.

    Supports round-robin and weighted sampling scheduling.

    Usage:
        scheduler = TaskScheduler(["flood", "burn", "lulc"])
        task = scheduler.next_task()  # Returns "flood"
        task = scheduler.next_task()  # Returns "burn"
        task = scheduler.next_task()  # Returns "lulc"
        task = scheduler.next_task()  # Returns "flood" (loops)
    """

    def __init__(
        self,
        tasks: List[str],
        weights: Optional[List[float]] = None,
    ):
        """Initialize task scheduler.

        Args:
            tasks: List of task names
            weights: Optional list of task weights (for weighted sampling)
        """
        self.tasks = tasks
        self.index = 0

        # Set up weights for weighted sampling
        if weights is None:
            self.weights = [1.0] * len(tasks)
        else:
            if len(weights) != len(tasks):
                raise ValueError(
                    f"Weights length ({len(weights)}) must match tasks length ({len(tasks)})"
                )
            self.weights = weights

    def next_task(self) -> str:
        """Get next task in round-robin fashion.

        Returns:
            Next task name
        """
        task = self.tasks[self.index]
        self.index = (self.index + 1) % len(self.tasks)
        return task

    def sample_task(self) -> str:
        """Sample a task based on weights.

        Returns:
            Sampled task name
        """
        return random.choices(self.tasks, weights=self.weights, k=1)[0]

    def reset(self) -> None:
        """Reset scheduler to beginning."""
        self.index = 0

    def current_task(self) -> str:
        """Get current task without advancing.

        Returns:
            Current task name
        """
        return self.tasks[self.index]

    def task_count(self) -> int:
        """Get number of tasks.

        Returns:
            Number of tasks
        """
        return len(self.tasks)

    def get_task_weights(self) -> Dict[str, float]:
        """Get task weights as dictionary.

        Returns:
            Dictionary mapping task name to weight
        """
        return dict(zip(self.tasks, self.weights))


class MultiTaskScheduler:
    """Advanced scheduler supporting task configurations.

    Usage:
        configs = [
            TaskConfig("flood", weight=1.0),
            TaskConfig("burn", weight=1.0),
            TaskConfig("lulc", weight=0.5),
        ]
        scheduler = MultiTaskScheduler(configs)
        task = scheduler.next_task()
    """

    def __init__(self, configs: List[TaskConfig]):
        """Initialize multi-task scheduler.

        Args:
            configs: List of TaskConfig objects
        """
        self.configs = configs
        self.index = 0

        # Filter enabled tasks
        self.enabled_tasks = [c.name for c in configs if c.enabled]
        self.weights = [c.weight for c in configs if c.enabled]

    def next_task(self) -> str:
        """Get next task in round-robin fashion.

        Returns:
            Next task name
        """
        task = self.enabled_tasks[self.index]
        self.index = (self.index + 1) % len(self.enabled_tasks)
        return task

    def sample_task(self) -> str:
        """Sample a task based on weights.

        Returns:
            Sampled task name
        """
        return random.choices(self.enabled_tasks, weights=self.weights, k=1)[0]

    def reset(self) -> None:
        """Reset scheduler to beginning."""
        self.index = 0

    def task_count(self) -> int:
        """Get number of enabled tasks.

        Returns:
            Number of enabled tasks
        """
        return len(self.enabled_tasks)


class BalanceScheduler(TaskScheduler):
    """Scheduler that balances tasks by keeping track of samples.

    Ensures tasks are sampled proportionally to their weights over time.
    """

    def __init__(
        self,
        tasks: List[str],
        weights: Optional[List[float]] = None,
    ):
        """Initialize balance scheduler.

        Args:
            tasks: List of task names
            weights: Optional list of task weights
        """
        super().__init__(tasks, weights)
        self.sample_counts = [0] * len(tasks)

    def next_task(self) -> str:
        """Get next task balancing sample counts.

        Returns:
            Task with lowest sample count relative to weight
        """
        # Calculate adjusted counts (count / weight)
        adjusted = [
            self.sample_counts[i] / self.weights[i]
            for i in range(len(self.tasks))
        ]

        # Find task with minimum adjusted count
        self.index = adjusted.index(min(adjusted))
        task = self.tasks[self.index]
        self.sample_counts[self.index] += 1

        return task


class PriorityScheduler(TaskScheduler):
    """Scheduler that respects task priorities.

    Higher priority tasks are sampled more frequently.
    """

    def __init__(
        self,
        tasks: List[str],
        priorities: List[float],
    ):
        """Initialize priority scheduler.

        Args:
            tasks: List of task names
            priorities: List of task priorities (higher = more frequent)
        """
        super().__init__(tasks)
        self.priorities = priorities

    def sample_task(self) -> str:
        """Sample task based on priorities.

        Returns:
            Sampled task name
        """
        return random.choices(self.tasks, weights=self.priorities, k=1)[0]


class SequentialScheduler(TaskScheduler):
    """Scheduler that runs tasks sequentially in batches.

    Runs N steps for each task before moving to the next.
    """

    def __init__(
        self,
        tasks: List[str],
        steps_per_task: int = 10,
    ):
        """Initialize sequential scheduler.

        Args:
            tasks: List of task names
            steps_per_task: Number of steps to run per task
        """
        super().__init__(tasks)
        self.steps_per_task = steps_per_task
        self.current_step = 0

    def next_task(self) -> str:
        """Get next task, advancing after steps_per_task iterations.

        Returns:
            Next task name
        """
        self.current_step += 1

        if self.current_step >= self.steps_per_task:
            self.index = (self.index + 1) % len(self.tasks)
            self.current_step = 0

        return self.tasks[self.index]

    def reset(self) -> None:
        """Reset scheduler to beginning."""
        super().reset()
        self.current_step = 0