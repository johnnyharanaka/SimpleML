"""ResNet model block backed by timm."""

from __future__ import annotations

from typing import Literal

import timm
import torch
from torch import nn

from simpleml.registries import MODELS

Variant = Literal["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]


@MODELS.register
class ResNet(nn.Module):
    """ResNet architecture selectable by variant, built via timm."""

    def __init__(
        self,
        variant: Variant = "resnet50",
        pretrained: bool = False,
        num_classes: int = 1000,
        in_chans: int = 3,
    ) -> None:
        super().__init__()
        self.model = timm.create_model(
            variant,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_chans,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @property
    def num_classes(self) -> int:
        return self.model.num_classes

    @property
    def num_features(self) -> int:
        return self.model.num_features
