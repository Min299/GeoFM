"""geofm.builders.model_builder

Builder for creating GeoFM models.
"""
from __future__ import annotations

import torch.nn as nn

from geofm.models.multitask.shared_geofm import SharedGeoFM


class ModelBuilder:
    """Builder for creating GeoFM models.

    Usage:
        builder = ModelBuilder()
        model = builder.build_shared_model(backbone, adapter_bank, decoder_bank)
    """

    def __init__(self):
        """Initialize model builder."""
        pass

    def build_shared_model(
        self,
        backbone: nn.Module,
        adapter_bank: nn.Module,
        decoder_bank: nn.Module,
    ) -> SharedGeoFM:
        """Build a shared GeoFM model.

        Args:
            backbone: Backbone encoder module
            adapter_bank: Bank of task adapters
            decoder_bank: Bank of task decoders

        Returns:
            SharedGeoFM model
        """
        return SharedGeoFM(
            backbone=backbone,
            adapter_bank=adapter_bank,
            decoder_bank=decoder_bank,
        )

    def build_from_config(self, config) -> SharedGeoFM:
        """Build model from experiment config.

        Args:
            config: ExperimentConfig object

        Returns:
            SharedGeoFM model

        Raises:
            NotImplementedError: Full config building not yet implemented
        """
        raise NotImplementedError("Config-based model building pending.")

    def get_model_info(self, model: nn.Module) -> dict:
        """Get information about a model.

        Args:
            model: PyTorch model

        Returns:
            Dictionary with model information
        """
        from geofm.utils.model_stats import get_model_summary

        summary = get_model_summary(model)

        return {
            "total_parameters": summary["total_parameters"],
            "trainable_parameters": summary["trainable_parameters"],
            "frozen_parameters": summary["frozen_parameters"],
            "trainable_percentage": summary["trainable_percentage"],
        }