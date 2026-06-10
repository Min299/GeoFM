"""geofm.experiments.metadata_dropper

Random metadata dropout for ablation studies.
"""
from __future__ import annotations

import random


class MetadataDropper:
    """Randomly drops metadata for ablation studies."""

    def __init__(
        self,
        drop_prob: float = 0.5,
    ):
        """Initialize dropper.

        Args:
            drop_prob: Probability of dropping metadata (0 to 1)
        """
        self.drop_prob = drop_prob

    def __call__(
        self,
        metadata: dict,
    ):
        """Apply dropout to metadata.

        Args:
            metadata: Input metadata dictionary

        Returns:
            Original metadata or empty dict based on drop probability
        """
        if random.random() > self.drop_prob:

            return metadata

        return {}