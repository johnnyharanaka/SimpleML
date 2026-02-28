"""CosineAnnealingLR scheduler wrapper registered for config-driven use."""

from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR as _CosineAnnealingLR

from simpleml.registries import SCHEDULERS


@SCHEDULERS.register
class CosineAnnealingLR(_CosineAnnealingLR):
    """Registry-friendly wrapper around ``CosineAnnealingLR``."""

    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int = 50,  # noqa: N803
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        super().__init__(
            optimizer,
            T_max=T_max,
            eta_min=eta_min,
            last_epoch=last_epoch,
        )
