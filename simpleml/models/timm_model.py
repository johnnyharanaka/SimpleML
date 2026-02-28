"""Timm model wrapper for loading any timm model by name."""

from __future__ import annotations

from typing import Any

import timm
import torch
from torch import nn

from simpleml.registries import MODELS


@MODELS.register
class TimmModel(nn.Module):
    """Registry-friendly wrapper around ``timm.create_model``.

    Delegates model creation to timm so that any architecture available in the
    timm library can be instantiated by name from a SimpleML config.
    """

    def __init__(
        self,
        model_name: str,
        pretrained: bool = False,
        num_classes: int = 1000,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass, delegating to the wrapped timm model."""
        return self.model(x)

    @property
    def num_classes(self) -> int:
        """Number of output classes."""
        return self.model.num_classes

    @property
    def num_features(self) -> int:
        """Number of features from the final pooling layer."""
        return self.model.num_features
