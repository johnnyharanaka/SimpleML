"""SGD optimizer wrapper registered for config-driven use."""

from __future__ import annotations

from torch.optim import SGD as _SGD

from simpleml.registries import OPTIMIZERS


@OPTIMIZERS.register
class SGD(_SGD):
    """Registry-friendly wrapper around ``torch.optim.SGD``."""

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
