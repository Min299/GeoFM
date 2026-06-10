"""geofm.metadata.metadata_validator

Metadata field validation.
"""
from __future__ import annotations

from geofm.metadata.metadata_schema import (
    DEFAULT_METADATA_FIELDS,
)


class MetadataValidator:
    """Validator for metadata field compliance."""

    @staticmethod
    def validate(
        metadata: dict,
    ):
        """Validate metadata fields against schema.

        Args:
            metadata: Dictionary of metadata fields

        Returns:
            True if valid

        Raises:
            ValueError: If unknown fields are present
        """
        unknown = set(
            metadata.keys()
        ) - DEFAULT_METADATA_FIELDS

        if unknown:

            raise ValueError(
                f"Unknown metadata fields: {unknown}"
            )

        return True