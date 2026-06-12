"""geofm.metadata.metadata_encoder

Converts metadata dictionary into tensor representation.
"""
from __future__ import annotations

import torch


class MetadataEncoder:
    """
    Converts metadata dictionary
    into tensor representation.
    """

    def encode(
        self,
        metadata: dict,
    ):
        """Encode metadata dict as float tensor.

        Args:
            metadata: Dictionary with metadata fields

        Returns:
            torch.Tensor of encoded values
        """
        values = []

        values.append(
            float(
                metadata.get(
                    "latitude",
                    0.0,
                )
            )
        )

        values.append(
            float(
                metadata.get(
                    "longitude",
                    0.0,
                )
            )
        )

        values.append(
            float(
                metadata.get(
                    "resolution",
                    0.0,
                )
            )
        )

        return torch.tensor(
            values,
            dtype=torch.float32,
        )
