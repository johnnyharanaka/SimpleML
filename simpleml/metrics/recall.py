"""Recall metric registered for config-driven use."""

from __future__ import annotations

import torch
from sklearn.metrics import recall_score
from torch import Tensor

from simpleml.metrics.base import Metric
from simpleml.registries import METRICS


@METRICS.register
class Recall(Metric):
    """Classification recall.

    Accepts logits/probabilities of shape ``(B, C)`` or pre-argmaxed
    predictions of shape ``(B,)``.
    """

    def __init__(self, average: str = "macro", zero_division: float = 0.0) -> None:
        super().__init__()
        self.average = average
        self.zero_division = zero_division

    def update(self, preds: Tensor, targets: Tensor) -> None:
        """Accumulate a batch, applying argmax if preds are 2-D."""
        if preds.ndim == 2:
            preds = preds.argmax(dim=1)
        self._preds.append(preds)
        self._targets.append(targets)

    def compute(self) -> float:
        """Return recall as a float via ``sklearn.metrics.recall_score``."""
        self._check_not_empty()
        y_pred = torch.cat(self._preds).cpu().numpy()
        y_true = torch.cat(self._targets).cpu().numpy()
        return float(
            recall_score(
                y_true,
                y_pred,
                average=self.average,
                zero_division=self.zero_division,
            )
        )
