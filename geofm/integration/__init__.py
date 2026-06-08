"""geofm.integration

Integration utilities for connecting backbones to GeoFM.

This module provides the contracts that all backbones must obey:
- BackboneAdapter: ABC for all backbones
- FeatureExtractorAdapter: Wrapper for feature extraction
- ShapeValidator: Validates feature map shapes
- ModelValidator: Validates model structure
"""
from geofm.integration.backbone_adapter import BackboneAdapter
from geofm.integration.feature_extractor_adapter import FeatureExtractorAdapter
from geofm.integration.shape_validator import ShapeValidator
from geofm.integration.model_validator import ModelValidator

__all__ = [
    "BackboneAdapter",
    "FeatureExtractorAdapter",
    "ShapeValidator",
    "ModelValidator",
]