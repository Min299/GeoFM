"""geofm.models.fusion

Fusion modules for combining image and metadata tokens.
"""
from geofm.models.fusion.base_fusion import BaseFusion
from geofm.models.fusion.concat_fusion import ConcatFusion
from geofm.models.fusion.film_fusion import FiLMFusion
from geofm.models.fusion.cross_attention_fusion import CrossAttentionFusion

__all__ = [
    "BaseFusion",
    "ConcatFusion",
    "FiLMFusion",
    "CrossAttentionFusion",
]