"""geofm.models.lora.task_bank

Task-specific LoRA adapter bank.
Crucial for Phase 2: Shared Backbone + Task LoRA Bank.

This file survives all phases of the research:
- Phase 1: LoRA vs Full FT comparison
- Phase 2: Shared backbone with task-specific adapters
- Phase 3+: Any multi-task scenario

Usage:
    bank = TaskLoRABank()

    # Register task adapters
    bank.register_task("flood", flood_lora)
    bank.register_task("burn", burn_lora)
    bank.register_task("crop", crop_lora)

    # Get adapter for inference
    flood_adapter = bank.get("flood")

    # List available tasks
    tasks = bank.tasks()
"""
from typing import Dict, Optional

import torch.nn as nn


class TaskLoRABank(nn.Module):
    """Bank of task-specific LoRA adapters.

    Stores and manages LoRA adapters for different tasks.
    Enables efficient multi-task inference by selecting
    the appropriate adapter at runtime.
    """

    def __init__(self):
        super().__init__()
        self.adapters = nn.ModuleDict()

    def register_task(self, task_name: str, adapter: nn.Module):
        """Register a LoRA adapter for a task."""
        self.adapters[task_name] = adapter

    def get(self, task_name: str) -> Optional[nn.Module]:
        """Get LoRA adapter for a task."""
        return self.adapters.get(task_name)

    def tasks(self) -> list:
        """Get list of registered task names."""
        return list(self.adapters.keys())

    def has_task(self, task_name: str) -> bool:
        """Check if task is registered."""
        return task_name in self.adapters

    def num_tasks(self) -> int:
        """Get number of registered tasks."""
        return len(self.adapters)

    def summary(self) -> Dict[str, int]:
        """Get summary of adapters with parameter counts."""
        return {
            task: sum(p.numel() for p in adapter.parameters())
            for task, adapter in self.adapters.items()
        }
