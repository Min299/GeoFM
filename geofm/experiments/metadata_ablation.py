"""geofm.experiments.metadata_ablation

Metadata ablation strategies for experiments.
"""
from __future__ import annotations


class MetadataAblation:
    """Applies ablation strategy to metadata."""

    def __init__(
        self,
        strategy,
    ):
        """Initialize with ablation strategy.

        Args:
            strategy: Callable that transforms metadata
        """
        self.strategy = strategy

    def apply(
        self,
        metadata,
    ):
        """Apply ablation strategy to metadata.

        Args:
            metadata: Input metadata dictionary

        Returns:
            Ablated metadata
        """
        return self.strategy(
            metadata
        )