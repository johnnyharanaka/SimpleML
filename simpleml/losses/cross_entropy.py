"""Cross-entropy loss wrapper registered for config-driven use."""

from __future__ import annotations

import torch
from torch import nn

from simpleml.registries import LOSSES


@LOSSES.register
class CrossEntropyLoss(nn.CrossEntropyLoss):
    """Registry-friendly wrapper around ``nn.CrossEntropyLoss``."""

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__(
            weight=weight,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction=reduction,
        )
