"""geofm.models.peft

Parameter-Efficient Fine-Tuning modules.

Includes:
- LoRA (Low-Rank Adaptation) for attention layers
- Feature Adapters for multi-scale feature processing
- Hybrid adapters combining multiple techniques
"""
from geofm.models.peft.feature_adapter import FeatureAdapter
from geofm.models.peft.task_feature_adapter import TaskFeatureAdapter
from geofm.models.peft.adapter_bank import AdapterBank

# LoRA modules
from geofm.models.peft.lora_layer import (
    LoRALinear,
    LoRAConfig,
    apply_lora_to_linear,
)
from geofm.models.peft.lora_adapter import (
    TerraMindLoRA,
    inject_lora,
    inject_lora_explicit,
    freeze_all_except_lora,
)
from geofm.models.peft.hybrid_adapter import (
    HybridAdapter,
    LoRAFeatureHybrid,
)
from geofm.models.peft.parameter_counter import (
    count_total_params,
    count_trainable_params,
    count_frozen_params,
    count_lora_params,
    count_lora_layers,
    trainable_ratio,
    peft_percentage,
    format_params,
    print_param_summary,
    verify_peft_ready,
    verify_lora_only_trainable,
    summary,
    ParameterCounter,
)

__all__ = [
    # Feature adapters (existing)
    "FeatureAdapter",
    "TaskFeatureAdapter",
    "AdapterBank",
    # LoRA layers
    "LoRALinear",
    "LoRAConfig",
    "apply_lora_to_linear",
    # LoRA adapter
    "TerraMindLoRA",
    "inject_lora",
    "inject_lora_explicit",
    "freeze_all_except_lora",
    # Hybrid
    "HybridAdapter",
    "LoRAFeatureHybrid",
    # Utilities
    "count_total_params",
    "count_trainable_params",
    "count_frozen_params",
    "count_lora_params",
    "count_lora_layers",
    "trainable_ratio",
    "peft_percentage",
    "format_params",
    "print_param_summary",
    "verify_peft_ready",
    "verify_lora_only_trainable",
    "summary",
    "ParameterCounter",
]