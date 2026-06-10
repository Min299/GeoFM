"""geofm.builders.fusion_builder

Builder for fusion modules.
"""
from __future__ import annotations

from geofm.models.fusion.concat_fusion import (
    ConcatFusion,
)

from geofm.models.fusion.film_fusion import (
    FiLMFusion,
)

from geofm.models.fusion.cross_attention_fusion import (
    CrossAttentionFusion,
)


class FusionBuilder:
    """Builder for creating fusion modules."""

    @staticmethod
    def build(
        name,
        embed_dim,
    ):
        """Build fusion module by name.

        Args:
            name: Fusion type ("concat", "film", "cross_attention")
            embed_dim: Embedding dimension

        Returns:
            Fusion module instance

        Raises:
            ValueError: If unknown fusion type
        """
        if name == "concat":

            return ConcatFusion()

        if name == "film":

            return FiLMFusion(
                embed_dim
            )

        if name == "cross_attention":

            return (
                CrossAttentionFusion(
                    embed_dim
                )
            )

        raise ValueError(
            f"Unknown fusion {name}"
        )