"""geofm.factories.decoder_factory

Factory for creating decoder modules.
"""
from __future__ import annotations

from geofm.models.decoders.flood_decoder import FloodDecoder
from geofm.models.decoders.burn_decoder import BurnDecoder
from geofm.models.decoders.lulc_decoder import LULCDecoder


class DecoderFactory:
    """Factory for creating decoder modules.

    Usage:
        decoder = DecoderFactory.build("flood")
    """

    @staticmethod
    def build(
        task: str,
        **kwargs,
    ):
        """Build a decoder for the specified task.

        Args:
            task: Task name (flood, burn, lulc)
            **kwargs: Decoder-specific arguments

        Returns:
            Decoder instance

        Raises:
            ValueError: If task is unknown
        """
        if task == "flood":
            return FloodDecoder(**kwargs)

        if task == "burn":
            return BurnDecoder(**kwargs)

        if task == "lulc":
            return LULCDecoder(**kwargs)

        raise ValueError(f"Unknown task: {task}")