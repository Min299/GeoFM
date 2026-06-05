"""geofm.datasets.flood.sen1floods11

Reference dataset implementation for Sen1Floods11. Every future dataset should
follow this same contract: ``dataset[idx]`` returns
``{"modalities", "metadata", "task", "label"}``.

Expected folder structure::

    data/sen1floods11/
        S2L1C/   *.tif  (Sentinel-2 images)
        S1GRD/   *.tif  (SAR images, optional)
        Label/   *.tif  (same filenames as S2L1C)
        metadata.csv    (columns: filename, latitude, longitude, day_of_year)
"""
from pathlib import Path

import rasterio
import torch
import pandas as pd

from geofm.datasets.base_dataset import BaseGeoDataset
from geofm.metadata.metadata_encoder import MetadataEncoder


class Sen1Floods11Dataset(BaseGeoDataset):
    def __init__(self, root_dir, metadata_csv=None, transform=None,
                 modalities=None):
        super().__init__(
            root_dir=root_dir,
            task="flood",
            modalities=modalities or ["S2L1C"]
        )

        self.transform = transform
        # Modality directories
        self.modality_dirs = {
            mod: self.root_dir / mod for mod in self.modalities
        }
        self.label_dir = self.root_dir / "Label"

        # Get sample list from first modality
        first_mod = self.modalities[0]
        self.samples = sorted(
            list(self.modality_dirs[first_mod].glob("*.tif"))
        )

        if metadata_csv:
            self.metadata_df = pd.read_csv(metadata_csv)
        else:
            self.metadata_df = None

    def __len__(self):
        return len(self.samples)

    def load_modalities(self, idx):
        """Load all modality tensors for a sample.

        Returns:
            dict: {"S2L1C": tensor, "S1GRD": tensor, ...}
        """
        image_path = self.samples[idx]
        modalities = {}

        for mod in self.modalities:
            mod_path = self.modality_dirs[mod] / image_path.name
            if mod_path.exists():
                with rasterio.open(mod_path) as src:
                    modalities[mod] = torch.tensor(
                        src.read(), dtype=torch.float32
                    )
            else:
                # Return zeros if modality not available
                modalities[mod] = torch.zeros(
                    13 if "S2" in mod else 2, 256, 256
                )

        return modalities

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
