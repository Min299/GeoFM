"""geofm.data.samplers

Task samplers for multi-task data loading.

Provides various sampling strategies for multi-task datasets.
"""
from __future__ import annotations

from typing import List, Optional
import random


class RoundRobinSampler:
    """Sampler that cycles through tasks in round-robin order.

    Usage:
        sampler = RoundRobinSampler(["flood", "burn", "lulc"])
        task = sampler.sample()  # Returns "flood"
        task = sampler.sample()  # Returns "burn"
        task = sampler.sample()  # Returns "lulc"
        task = sampler.sample()  # Returns "flood"
    """

    def __init__(
        self,
        tasks: List[str],
    ):
        """Initialize round-robin sampler.

        Args:
            tasks: List of task names
        """
        self.tasks = tasks
        self.idx = 0

    def sample(self) -> str:
        """Get next task in round-robin order.

        Returns:
            Task name
        """
        task = self.tasks[self.idx]
        self.idx = (self.idx + 1) % len(self.tasks)
        return task

    def reset(self) -> None:
        """Reset sampler to beginning."""
        self.idx = 0

    def current_task(self) -> str:
        """Get current task without advancing.

        Returns:
            Current task name
        """
        return self.tasks[self.idx]

    def task_count(self) -> int:
        """Get number of tasks.

        Returns:
            Number of tasks
        """
        return len(self.tasks)


class RandomTaskSampler:
    """Sampler that randomly selects tasks.

    Usage:
        sampler = RandomTaskSampler(["flood", "burn"])
        task = sampler.sample()  # Returns random task
    """

    def __init__(
        self,
        tasks: List[str],
        weights: Optional[List[float]] = None,
    ):
        """Initialize random sampler.

        Args:
            tasks: List of task names
            weights: Optional sampling weights
        """
        self.tasks = tasks
        self.weights = weights or [1.0] * len(tasks)

    def sample(self) -> str:
        """Get random task based on weights.

        Returns:
            Task name
        """
        return random.choices(self.tasks, weights=self.weights, k=1)[0]

    def sample_batch(self, batch_size: int) -> List[str]:
        """Sample multiple tasks.

        Args:
            batch_size: Number of tasks to sample

        Returns:
            List of task names
        """
        return random.choices(self.tasks, weights=self.weights, k=batch_size)


class WeightedTaskSampler:
    """Sampler that respects task weights for sampling frequency.

    Usage:
        sampler = WeightedTaskSampler(
            tasks=["flood", "burn", "lulc"],
            weights=[2.0, 1.0, 1.0],
        )
        # Flood will be sampled twice as often
    """

    def __init__(
        self,
        tasks: List[str],
        weights: List[float],
    ):
        """Initialize weighted sampler.

        Args:
            tasks: List of task names
            weights: Sampling weights (higher = more frequent)
        """
        if len(weights) != len(tasks):
            raise ValueError("Weights length must match tasks length")

        self.tasks = tasks
        self.weights = weights

    def sample(self) -> str:
        """Sample task based on weights.

        Returns:
            Task name
        """
        return random.choices(self.tasks, weights=self.weights, k=1)[0]

    def get_weight(self, task: str) -> float:
        """Get weight for a task.

        Args:
            task: Task name

        Returns:
            Weight value
        """
        if task in self.tasks:
            return self.weights[self.tasks.index(task)]
        return 0.0


class PriorityTaskSampler:
    """Sampler that uses task priorities for sampling.

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
            priorities: Task priorities
        """
        self.tasks = tasks
        self.priorities = priorities

    def sample(self) -> str:
        """Sample task based on priority.

        Returns:
            Task name
        """
        return random.choices(self.tasks, weights=self.priorities, k=1)[0]

    def set_priority(self, task: str, priority: float) -> None:
        """Update priority for a task.

        Args:
            task: Task name
            priority: New priority value
        """
        if task in self.tasks:
            self.priorities[self.tasks.index(task)] = priority


class BalancedTaskSampler:
    """Sampler that maintains balance across tasks.

    Ensures each task contributes equally to training.
    """

    def __init__(
        self,
        tasks: List[str],
        target_counts: Optional[dict] = None,
    ):
        """Initialize balanced sampler.

        Args:
            tasks: List of task names
            target_counts: Optional target sample counts per task
        """
        self.tasks = tasks
        self.target_counts = target_counts or {task: 0 for task in tasks}
        self.current_counts = {task: 0 for task in tasks}

    def sample(self) -> str:
        """Sample task maintaining balance.

        Returns:
            Task name
        """
        # Find tasks with lowest current count relative to target
        min_count = float("inf")
        candidates = []

        for task in self.tasks:
            current = self.current_counts.get(task, 0)
            target = self.target_counts.get(task, 0)

            # Calculate ratio or use absolute count if no target
            if target > 0:
                ratio = current / target
            else:
                ratio = current

            if ratio < min_count:
                min_count = ratio
                candidates = [task]
            elif ratio == min_count:
                candidates.append(task)

        # Choose randomly from candidates
        task = random.choice(candidates)
        self.current_counts[task] += 1

        return task

    def reset_counts(self) -> None:
        """Reset current counts."""
        self.current_counts = {task: 0 for task in self.tasks}