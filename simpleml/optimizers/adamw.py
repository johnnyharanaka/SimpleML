"""AdamW optimizer wrapper registered for config-driven use."""

from __future__ import annotations

from torch.optim import AdamW as _AdamW

from simpleml.registries import OPTIMIZERS


@OPTIMIZERS.register
class AdamW(_AdamW):
    """Registry-friendly wrapper around ``torch.optim.AdamW``."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
    ) -> None:
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
