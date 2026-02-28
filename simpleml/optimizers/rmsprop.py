"""RMSprop optimizer wrapper registered for config-driven use."""

from __future__ import annotations

from torch.optim import RMSprop as _RMSprop

from simpleml.registries import OPTIMIZERS


@OPTIMIZERS.register
class RMSprop(_RMSprop):
    """Registry-friendly wrapper around ``torch.optim.RMSprop``."""

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        centered: bool = False,
    ) -> None:
        super().__init__(
            params,
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered,
        )
