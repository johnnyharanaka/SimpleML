"""Adam optimizer wrapper registered for config-driven use."""

from __future__ import annotations

from torch.optim import Adam as _Adam

from simpleml.registries import OPTIMIZERS


@OPTIMIZERS.register
class Adam(_Adam):
    """Registry-friendly wrapper around ``torch.optim.Adam``."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
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
