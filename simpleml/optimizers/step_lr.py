"""StepLR scheduler wrapper registered for config-driven use."""

from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR as _StepLR

from simpleml.registries import SCHEDULERS


@SCHEDULERS.register
class StepLR(_StepLR):
    """Registry-friendly wrapper around ``torch.optim.lr_scheduler.StepLR``."""

    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int = 10,
        gamma: float = 0.1,
        last_epoch: int = -1,
    ) -> None:
        super().__init__(
            optimizer,
            step_size=step_size,
            gamma=gamma,
            last_epoch=last_epoch,
        )
