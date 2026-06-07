"""geofm.models.peft

Parameter-Efficient Fine-Tuning modules.

Note: These are Feature Adapters (applied to extracted features),
not LoRA (which modifies attention projections inside the backbone).
"""
from geofm.models.peft.feature_adapter import FeatureAdapter

__all__ = [
    "FeatureAdapter",
]