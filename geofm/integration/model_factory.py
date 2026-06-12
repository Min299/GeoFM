"""geofm.integration.model_factory

Factory for building GeoFM models from configs.
"""
from __future__ import annotations

from geofm.builders.model_builder import (
    ModelBuilder,
)


class ModelFactory:
    """Factory for creating GeoFM models."""

    @staticmethod
    def build(
        cfg,
    ):
        """Build model from config.

        Args:
            cfg: Configuration object

        Returns:
            Built model
        """
        return (
            ModelBuilder(
                cfg
            ).build()
        )