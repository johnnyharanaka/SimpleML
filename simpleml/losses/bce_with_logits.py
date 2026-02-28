"""Binary cross-entropy with logits loss wrapper registered for config-driven use."""

from __future__ import annotations

import torch
from torch import nn

from simpleml.registries import LOSSES


@LOSSES.register
class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    """Registry-friendly wrapper around ``nn.BCEWithLogitsLoss``."""

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        pos_weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(
            weight=weight,
            pos_weight=pos_weight,
            reduction=reduction,
        )
