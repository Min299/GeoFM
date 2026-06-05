"""Contract test for the Sen1Floods11 reference dataset.

Validates that ``dataset[idx]`` yields the uniform sample dict
``{"modalities", "metadata", "task", "label"}``. Skips gracefully when the
optional heavy dependencies or the dataset files are not available.
"""
from pathlib import Path

import pytest

# Heavy/optional deps -- skip the whole module if any is missing.
pytest.importorskip("torch")
pytest.importorskip("rasterio")
pytest.importorskip("pandas")

from geofm.datasets.flood.sen1floods11 import Sen1Floods11Dataset

DATA_ROOT = Path("data/sen1floods11")


@pytest.mark.skipif(
    not (DATA_ROOT / "S2L1C").is_dir(),
    reason="sen1floods11 data not present at data/sen1floods11/",
)
def test_flood_sample_contract():
    dataset = Sen1Floods11Dataset(root_dir=str(DATA_ROOT))

    assert len(dataset) > 0, "no .tif samples found under S2L1C/"

    sample = dataset[0]

    # New contract: modalities dict instead of image
    assert set(sample.keys()) == {"modalities", "metadata", "task", "label"}
    assert sample["task"] == "flood"
    assert isinstance(sample["modalities"], dict), "modalities must be a dict"
    assert "S2L1C" in sample["modalities"], "S2L1C modality required"
    assert sample["modalities"]["S2L1C"].ndim == 3  # (C, H, W)
    assert sample["label"].ndim == 2  # (H, W)
