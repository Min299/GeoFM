"""geofm.builders.shared_model_builder

Central construction entrypoint for building SharedGeoFM models.
"""
from __future__ import annotations

from typing import Optional, Dict, Any

from geofm.models.multitask.shared_geofm import SharedGeoFM


class SharedModelBuilder:
    """Builder for creating SharedGeoFM models.

    This is the central construction entrypoint that assembles
    all components (backbone, adapter_bank, decoder_bank) into
    a complete model.

    Usage:
        builder = SharedModelBuilder(
            backbone=backbone,
            adapter_bank=adapter_bank,
            decoder_bank=decoder_bank,
        )
        model = builder.build()
    """

    def __init__(
        self,
        backbone: Optional[Any] = None,
        adapter_bank: Optional[Any] = None,
        decoder_bank: Optional[Any] = None,
    ):
        """Initialize the builder.

        Args:
            backbone: Backbone encoder module
            adapter_bank: Bank of task adapters
            decoder_bank: Bank of task decoders
        """
        self.backbone = backbone
        self.adapter_bank = adapter_bank
        self.decoder_bank = decoder_bank

        self._config = {}

    def set_backbone(self, backbone: Any) -> "SharedModelBuilder":
        """Set the backbone.

        Args:
            backbone: Backbone module

        Returns:
            Self for chaining
        """
        self.backbone = backbone
        return self

    def set_adapter_bank(self, adapter_bank: Any) -> "SharedModelBuilder":
        """Set the adapter bank.

        Args:
            adapter_bank: Adapter bank module

        Returns:
            Self for chaining
        """
        self.adapter_bank = adapter_bank
        return self

    def set_decoder_bank(self, decoder_bank: Any) -> "SharedModelBuilder":
        """Set the decoder bank.

        Args:
            decoder_bank: Decoder bank module

        Returns:
            Self for chaining
        """
        self.decoder_bank = decoder_bank
        return self

    def set_config(self, config: Dict[str, Any]) -> "SharedModelBuilder":
        """Set configuration dictionary.

        Args:
            config: Configuration options

        Returns:
            Self for chaining
        """
        self._config = config
        return self

    def build(self) -> SharedGeoFM:
        """Build the SharedGeoFM model.

        Returns:
            Assembled SharedGeoFM model

        Raises:
            ValueError: If required components are missing
        """
        if self.backbone is None:
            raise ValueError("Backbone must be set")

        if self.adapter_bank is None:
            raise ValueError("Adapter bank must be set")

        if self.decoder_bank is None:
            raise ValueError("Decoder bank must be set")

        return SharedGeoFM(
            backbone=self.backbone,
            adapter_bank=self.adapter_bank,
            decoder_bank=self.decoder_bank,
        )

    def build_with_lora(
        self,
        lora_rank: int = 16,
        lora_alpha: int = 32,
    ) -> SharedGeoFM:
        """Build model with LoRA injection.

        Args:
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha

        Returns:
            Model with LoRA injected
        """
        from geofm.models.peft.lora_adapter import inject_lora

        # First build the model
        model = self.build()

        # Inject LoRA into backbone
        if hasattr(self.backbone, '_model'):
            inject_lora(self.backbone._model, lora_rank, lora_alpha)

        return model

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model being built.

        Returns:
            Dictionary with model information
        """
        info = {
            "has_backbone": self.backbone is not None,
            "has_adapter_bank": self.adapter_bank is not None,
            "has_decoder_bank": self.decoder_bank is not None,
            "config": self._config,
        }

        if self.backbone is not None:
            info["backbone_type"] = type(self.backbone).__name__

        if self.adapter_bank is not None:
            info["adapter_bank_type"] = type(self.adapter_bank).__name__

        if self.decoder_bank is not None:
            info["decoder_bank_type"] = type(self.decoder_bank).__name__

        return info