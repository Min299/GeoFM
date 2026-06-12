"""geofm.research.fusion

Fusion modules for combining image and metadata tokens.
Archived: TerraMind handles multimodal fusion internally.
These are research experiments for future exploration.
"""
from geofm.research.fusion.base_fusion import BaseFusion
from geofm.research.fusion.concat_fusion import ConcatFusion
from geofm.research.fusion.film_fusion import FiLMFusion
from geofm.research.fusion.cross_attention_fusion import CrossAttentionFusion

__all__ = [
    "BaseFusion",
    "ConcatFusion",
    "FiLMFusion",
    "CrossAttentionFusion",
]