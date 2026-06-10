"""geofm.metadata.metadata_sample

Standardized metadata representation.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MetadataSample:
    """
    Standardized metadata representation.
    """

    latitude: float | None = None

    longitude: float | None = None

    timestamp: str | None = None

    sensor: str | None = None

    platform: str | None = None

    resolution: float | None = None

    task: str | None = None