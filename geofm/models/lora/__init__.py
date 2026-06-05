"""geofm.models.lora

LoRA module for efficient fine-tuning.
"""
from geofm.models.lora.config import LoRAConfig, LORA_CONFIGS
from geofm.models.lora.task_bank import TaskLoRABank
from geofm.models.lora.inject import (
    inject_lora,
    inject_lora_from_config,
    inject_task_lora,
    print_trainable_params,
)

__all__ = [
    "LoRAConfig",
    "LORA_CONFIGS",
    "TaskLoRABank",
    "inject_lora",
    "inject_lora_from_config",
    "inject_task_lora",
    "print_trainable_params",
]
