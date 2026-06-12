"""geofm.models.peft

Parameter-Efficient Fine-Tuning modules.

Includes:
- LoRA (Low-Rank Adaptation) for attention layers
- Feature Adapters for multi-scale feature processing
- Hybrid adapters combining multiple techniques
- PEFT wrappers for easy strategy selection
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

# LoRA adapter (standalone LoRA implementation)
try:
    from geofm.models.peft.lora_adapter import (
        LoRALinear as LoRALinearAdapter,
        LoRAConv2d,
        count_lora_parameters,
    )
except ImportError:
    LoRALinearAdapter = None
    LoRAConv2d = None
    count_lora_parameters = None

# TerraMindLoRA alias for backward compatibility
TerraMindLoRA = LoRALinearAdapter

# Hybrid adapters
try:
    from geofm.models.peft.hybrid_adapter import (
        HybridAdapter,
        LoRAFeatureHybrid,
    )
except ImportError:
    HybridAdapter = None
    LoRAFeatureHybrid = None

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

# PEFT wrappers
from geofm.models.peft.lora_wrapper import (
    LoRAWrapper,
)
from geofm.models.peft.adapter_wrapper import (
    AdapterWrapper,
)
from geofm.models.peft.peft_factory import (
    PEFTFactory,
)

__all__ = [
    # Feature adapters
    "FeatureAdapter",
    "TaskFeatureAdapter",
    "AdapterBank",
    # LoRA layers
    "LoRALinear",
    "LoRAConfig",
    "apply_lora_to_linear",
    # LoRA adapter
    "LoRALinearAdapter",
    "LoRAConv2d",
    "count_lora_parameters",
    # TerraMindLoRA alias
    "TerraMindLoRA",
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
    # PEFT wrappers
    "LoRAWrapper",
    "AdapterWrapper",
    "PEFTFactory",
]
