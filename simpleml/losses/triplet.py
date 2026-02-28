"""Triplet margin loss wrapper registered for config-driven use."""

from __future__ import annotations

from torch import nn

from simpleml.registries import LOSSES


@LOSSES.register
class TripletMarginLoss(nn.TripletMarginLoss):
    """Registry-friendly wrapper around ``nn.TripletMarginLoss``."""

    def __init__(
        self,
        margin: float = 1.0,
        p: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__(margin=margin, p=p, reduction=reduction)
