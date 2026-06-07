from __future__ import annotations


import torch
import torch.nn as nn




class ClassificationHead(nn.Module):


    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout: float = 0.1,
    ):
        super().__init__()


        self.pool = nn.AdaptiveAvgPool2d(1)


        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(
                in_channels,
                num_classes,
            ),
        )


    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:


        x = self.pool(x)


        return self.classifier(x)