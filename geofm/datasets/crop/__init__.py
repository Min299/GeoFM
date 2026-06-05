"""geofm.datasets.crop

Crop classification datasets.
"""
from geofm.datasets.crop.cropharvest import CropHarvestDataset
from geofm.datasets.crop.breizhcrops import BreizhCropsDataset
from geofm.datasets.crop.pastis import PastisDataset

__all__ = ["CropHarvestDataset", "BreizhCropsDataset", "PastisDataset"]
