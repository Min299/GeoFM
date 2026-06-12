"""geofm.metadata

Geospatial metadata utilities: GeoTIFF reading, coordinate parsing,
timestamp parsing, cyclic encoding, and standardized metadata representation.
"""
from geofm.metadata.geotiff_reader import GeoTIFFReader
from geofm.metadata.coordinate_parser import CoordinateParser
from geofm.metadata.timestamp_parser import TimestampParser
from geofm.metadata.metadata_encoder import MetadataEncoder
from geofm.metadata.metadata_sample import MetadataSample
from geofm.metadata.metadata_schema import DEFAULT_METADATA_FIELDS
from geofm.metadata.metadata_registry import MetadataRegistry
from geofm.metadata.metadata_validator import MetadataValidator
from geofm.metadata.metadata_builder import MetadataBuilder

__all__ = [
    # Readers
    "GeoTIFFReader",
    "CoordinateParser",
    "TimestampParser",
    # Encoding
    "MetadataEncoder",
    # Core
    "MetadataSample",
    "DEFAULT_METADATA_FIELDS",
    "MetadataRegistry",
    "MetadataValidator",
    "MetadataBuilder",
]
