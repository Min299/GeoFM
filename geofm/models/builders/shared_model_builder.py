from __future__ import annotations

from geofm.models.multitask.shared_geofm import (
    SharedGeoFM,
)

from geofm.models.peft.adapter_bank import (
    AdapterBank,
)

from geofm.models.decoders.decoder_bank import (
    DecoderBank,
)


def build_shared_model(
    backbone,
    adapters: dict,
    decoders: dict,
):
    """
    Construct:

    SharedGeoFM

        backbone

            +

        AdapterBank

            +

        DecoderBank
    """

    adapter_bank = AdapterBank()

    for (
        task_name,
        adapter,
    ) in adapters.items():

        adapter_bank.register_task(
            task_name,
            adapter,
        )

    decoder_bank = DecoderBank()

    for (
        task_name,
        decoder,
    ) in decoders.items():

        decoder_bank.register_task(
            task_name,
            decoder,
        )

    return SharedGeoFM(
        backbone=backbone,
        adapter_bank=adapter_bank,
        decoder_bank=decoder_bank,
    )