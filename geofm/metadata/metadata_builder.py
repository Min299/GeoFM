"""geofm.metadata.metadata_builder

Builder for MetadataSample from dictionary.
"""
from __future__ import annotations

from geofm.metadata.metadata_sample import (
    MetadataSample,
)


class MetadataBuilder:
    """Builder for creating MetadataSample from dictionaries."""

    @staticmethod
    def from_dict(
        data: dict,
    ):
        """Create MetadataSample from dictionary.

        Args:
            data: Dictionary with metadata fields

        Returns:
            MetadataSample instance
        """
        return MetadataSample(
            latitude=data.get(
                "latitude"
            ),
            longitude=data.get(
                "longitude"
            ),
            timestamp=data.get(
                "timestamp"
            ),
            sensor=data.get(
                "sensor"
            ),
            platform=data.get(
                "platform"
            ),
            resolution=data.get(
                "resolution"
            ),
            task=data.get(
                "task"
            ),
        )