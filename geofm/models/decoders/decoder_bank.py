from __future__ import annotations


import torch.nn as nn




class DecoderBank(nn.Module):


    def __init__(self):
        super().__init__()


        self.decoders = nn.ModuleDict()


    def register_task(
        self,
        task_name: str,
        decoder: nn.Module,
    ):
        self.decoders[
            task_name
        ] = decoder


    def get_decoder(
        self,
        task_name: str,
    ):
        return self.decoders[
            task_name
        ]


    def forward(
        self,
        task_name,
        features,
    ):
        decoder = self.get_decoder(
            task_name
        )


        return decoder(features)