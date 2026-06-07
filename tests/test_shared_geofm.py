import torch
import torch.nn as nn

from geofm.models.multitask.shared_geofm import (
    SharedGeoFM,
)


class DummyBackbone(
    nn.Module
):
    def forward(
        self,
        x,
    ):

        return [
            torch.randn(
                x.shape[0],
                256,
                64,
                64,
            )
            for _ in range(4)
        ]


class DummyAdapterBank(
    nn.Module
):
    def forward(
        self,
        task_name,
        features,
    ):
        return features


class DummyDecoderBank(
    nn.Module
):
    def forward(
        self,
        task_name,
        features,
    ):
        return torch.randn(
            features[0].shape[0],
            2,
            64,
            64,
        )


def test_shared_geofm():

    model = SharedGeoFM(
        backbone=DummyBackbone(),
        adapter_bank=DummyAdapterBank(),
        decoder_bank=DummyDecoderBank(),
    )

    x = torch.randn(
        2,
        3,
        224,
        224,
    )

    y = model(
        x,
        "flood",
    )

    assert y.shape == (
        2,
        2,
        64,
        64,
    )


def test_shared_geofm_routing():
    """Test that all tasks route correctly."""
    model = SharedGeoFM(
        backbone=DummyBackbone(),
        adapter_bank=DummyAdapterBank(),
        decoder_bank=DummyDecoderBank(),
    )

    x = torch.randn(2, 3, 224, 224)

    for task in ["flood", "burn", "lulc"]:
        y = model(x, task)
        assert y.shape == (2, 2, 64, 64), f"Failed for {task}"

    print("All routing tests passed ✅")


if __name__ == "__main__":
    test_shared_geofm()
    test_shared_geofm_routing()
    print("SharedGeoFM tests complete ✅")