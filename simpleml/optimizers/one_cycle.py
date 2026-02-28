"""OneCycleLR scheduler wrapper registered for config-driven use."""

from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR as _OneCycleLR

from simpleml.registries import SCHEDULERS


@SCHEDULERS.register
class OneCycleLR(_OneCycleLR):
    """Registry-friendly wrapper around ``torch.optim.lr_scheduler.OneCycleLR``."""

    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float | list[float] = 0.1,
        total_steps: int | None = None,
        epochs: int | None = None,
        steps_per_epoch: int | None = None,
        pct_start: float = 0.3,
        anneal_strategy: str = "cos",
        last_epoch: int = -1,
    ) -> None:
        super().__init__(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            last_epoch=last_epoch,
        )
