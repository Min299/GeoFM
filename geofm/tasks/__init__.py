"""geofm.tasks

Task utilities for multi-task learning.

Includes:
- task_scheduler: Scheduling strategies for task selection
- task_sampler: Sampling strategies for task selection
- task_batcher: Batching utilities for multiple task data loaders
"""
from geofm.tasks.task_scheduler import (
    TaskScheduler,
    MultiTaskScheduler,
    BalanceScheduler,
    PriorityScheduler,
    SequentialScheduler,
    TaskConfig,
)

from geofm.tasks.task_sampler import (
    TaskSampler,
    TaskSamplerWithRepetition,
    StratifiedSampler,
    PrioritySampler,
    AdaptiveSampler,
    CurriculumSampler,
)

from geofm.tasks.task_batcher import (
    TaskBatcher,
    TaskBatch,
    DynamicBatcher,
    MultiTaskBatch,
    RoundRobinBatcher,
)


__all__ = [
    # Scheduler
    "TaskScheduler",
    "MultiTaskScheduler",
    "BalanceScheduler",
    "PriorityScheduler",
    "SequentialScheduler",
    "TaskConfig",
    # Sampler
    "TaskSampler",
    "TaskSamplerWithRepetition",
    "StratifiedSampler",
    "PrioritySampler",
    "AdaptiveSampler",
    "CurriculumSampler",
    # Batcher
    "TaskBatcher",
    "TaskBatch",
    "DynamicBatcher",
    "MultiTaskBatch",
    "RoundRobinBatcher",
]