"""Abstract base class for all stateful metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor


class Metric(ABC):
    """Stateful metric that accumulates predictions across batches.

    Subclasses must implement ``update`` and ``compute``. The base class
    provides ``reset`` (clears accumulated state) and ``__call__`` (reset,
    update, compute in one shot).
    """

    def __init__(self) -> None:
        self._preds: list[Tensor] = []
        self._targets: list[Tensor] = []

    @abstractmethod
    def update(self, preds: Tensor, targets: Tensor) -> None:
        """Accumulate a batch of predictions and targets."""

    @abstractmethod
    def compute(self) -> Tensor | float:
        """Compute the metric from accumulated state."""

    def reset(self) -> None:
        """Clear accumulated predictions and targets."""
        self._preds.clear()
        self._targets.clear()

    def __call__(self, preds: Tensor, targets: Tensor) -> Tensor | float:
        """Reset, update with the given batch, and compute the metric."""
        self.reset()
        self.update(preds, targets)
        return self.compute()

    def _check_not_empty(self) -> None:
        """Raise if no data has been accumulated."""
        if len(self._preds) == 0:
            raise RuntimeError(
                f"{type(self).__name__}.compute() called before any update()."
            )
