"""geofm.experiments.metadata_masker

Masks metadata values for ablation studies.
"""
from __future__ import annotations


class MetadataMasker:
    """Masks metadata values to a constant."""

    def __init__(
        self,
        mask_value=0,
    ):
        """Initialize masker.

        Args:
            mask_value: Value to use for masking
        """
        self.mask_value = mask_value

    def __call__(
        self,
        metadata,
    ):
        """Mask all metadata values.

        Args:
            metadata: Input metadata dictionary

        Returns:
            Dictionary with all values set to mask_value
        """
        return {
            k: self.mask_value
            for k in metadata
        }