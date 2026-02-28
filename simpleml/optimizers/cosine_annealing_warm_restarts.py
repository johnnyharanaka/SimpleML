"""CosineAnnealingWarmRestarts scheduler wrapper registered for config-driven use."""

from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts as _CosineAnnealingWarmRestarts,
)

from simpleml.registries import SCHEDULERS


@SCHEDULERS.register
class CosineAnnealingWarmRestarts(_CosineAnnealingWarmRestarts):
    """Registry-friendly wrapper around ``CosineAnnealingWarmRestarts``."""

    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int = 10,  # noqa: N803
        T_mult: int = 1,  # noqa: N803
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        super().__init__(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min,
            last_epoch=last_epoch,
        )
