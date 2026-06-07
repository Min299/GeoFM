from __future__ import annotations


import torch
import torch.nn as nn
import torch.nn.functional as F




class PyramidDecoder(nn.Module):


    def __init__(
        self,
        in_channels,
        decoder_channels=256,
    ):
        super().__init__()


        self.lateral = nn.ModuleList(
            [
                nn.Conv2d(
                    c,
                    decoder_channels,
                    kernel_size=1,
                )
                for c in in_channels
            ]
        )


        self.output_conv = nn.Sequential(
            nn.Conv2d(
                decoder_channels,
                decoder_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
        )


    def forward(
        self,
        features,
    ):
        """
        features:
            [F2,F5,F8,F11]
        """


        feats = [
            layer(feature)
            for layer, feature
            in zip(
                self.lateral,
                features,
            )
        ]


        fused = feats[-1]


        for feat in reversed(feats[:-1]):


            fused = F.interpolate(
                fused,
                size=feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )


            fused = fused + feat


        return self.output_conv(
            fused
        )