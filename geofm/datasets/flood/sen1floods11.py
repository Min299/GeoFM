"""geofm.datasets.flood.sen1floods11

Reference dataset implementation for Sen1Floods11. Every future dataset should
follow this same contract: ``dataset[idx]`` returns
``{"image", "metadata", "task", "label"}``.

Expected folder structure::

    data/sen1floods11/
        images/   *.tif
        labels/   *.tif   (same filenames as images)
        metadata.csv       (columns: filename, latitude, longitude, day_of_year)
"""
from pathlib import Path

import rasterio
import torch
import pandas as pd

from geofm.datasets.base_dataset import BaseGeoDataset
from geofm.metadata.metadata_encoder import MetadataEncoder


class Sen1Floods11Dataset(BaseGeoDataset):
    def __init__(self, root_dir, metadata_csv=None, transform=None):
        super().__init__(root_dir=root_dir, task="flood")

        self.root_dir = Path(root_dir)
        self.transform = transform

        self.image_dir = self.root_dir / "images"
        self.label_dir = self.root_dir / "labels"

        self.samples = sorted(list(self.image_dir.glob("*.tif")))

        if metadata_csv:
            self.metadata_df = pd.read_csv(metadata_csv)
        else:
            self.metadata_df = None

    def __len__(self):
        return len(self.samples)

    def load_image(self, idx):
        image_path = self.samples[idx]

        with rasterio.open(image_path) as src:
            image = src.read()

        image = torch.tensor(image, dtype=torch.float32)
        return image

    def load_label(self, idx):
        image_path = self.samples[idx]
        label_path = self.label_dir / image_path.name

        with rasterio.open(label_path) as src:
            label = src.read(1)

        label = torch.tensor(label, dtype=torch.long)
        return label

    def load_metadata(self, idx):
        image_path = self.samples[idx]

        if self.metadata_df is not None:
            row = self.metadata_df[
                self.metadata_df["filename"] == image_path.name
            ].iloc[0]

            return MetadataEncoder.encode(
                lat=row["latitude"],
                lon=row["longitude"],
                day=row["day_of_year"],
            )

        return None
