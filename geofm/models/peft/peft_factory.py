"""geofm.models.peft.peft_factory

Factory for PEFT strategy selection.
"""
from __future__ import annotations

from geofm.models.peft.lora_wrapper import (
    LoRAWrapper,
)

from geofm.models.peft.adapter_wrapper import (
    AdapterWrapper,
)


class PEFTFactory:
    """Factory for creating PEFT-wrapped models.

    Usage:
        model = PEFTFactory.build("lora", base_model)
        model = PEFTFactory.build("adapter", base_model)
    """

    @staticmethod
    def build(
        strategy,
        model,
    ):
        """Build PEFT-wrapped model.

        Args:
            strategy: PEFT strategy ("lora" or "adapter")
            model: Base model to wrap

        Returns:
            PEFT-wrapped model, or original if no strategy matches
        """
        if strategy == "lora":

            return LoRAWrapper(
                model
            )

        if strategy == "adapter":

            return AdapterWrapper(
                model
            )

        # Return original model if no strategy matches
        return model