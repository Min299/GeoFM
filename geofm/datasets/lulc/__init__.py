"""geofm.datasets.lulc

Land use/land cover classification datasets.
"""
from geofm.datasets.lulc.bigearthnet import BigEarthNetDataset
from geofm.datasets.lulc.dynamicworld import DynamicWorldDataset

__all__ = ["BigEarthNetDataset", "DynamicWorldDataset"]
