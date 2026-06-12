"""geofm.factories.adapter_factory

Factory for creating adapter modules.
"""
from __future__ import annotations

from geofm.models.peft.feature_adapter import FeatureAdapter
from geofm.models.peft.task_feature_adapter import TaskFeatureAdapter


class AdapterFactory:
    """Factory for creating adapter modules.

    Usage:
        adapter = AdapterFactory.build("feature", dim=768, bottleneck_dim=64)
    """

    @staticmethod
    def build(
        adapter_type: str,
        **kwargs,
    ):
        """Build an adapter of the specified type.

        Args:
            adapter_type: Type of adapter (feature, task_feature, lora, hybrid)
            **kwargs: Adapter-specific arguments

        Returns:
            Adapter instance

        Raises:
            ValueError: If adapter type is unknown
            NotImplementedError: If adapter is not yet implemented
        """
        if adapter_type == "feature":
            return FeatureAdapter(**kwargs)

        elif adapter_type == "task_feature":
            return TaskFeatureAdapter(**kwargs)

        elif adapter_type == "lora":
            raise NotImplementedError("LoRA integration pending.")

        elif adapter_type == "hybrid":
            raise NotImplementedError("Hybrid integration pending.")

        elif adapter_type == "full_ft":
            # Full fine-tuning uses no adapter
            return None

        raise ValueError(f"Unknown adapter type: {adapter_type}")