"""Confusion matrix metric registered for config-driven use."""

from __future__ import annotations

import torch
from sklearn.metrics import confusion_matrix
from torch import Tensor

from simpleml.metrics.base import Metric
from simpleml.registries import METRICS


@METRICS.register
class ConfusionMatrix(Metric):
    """Confusion matrix for classification.

    Accepts logits/probabilities of shape ``(B, C)`` or pre-argmaxed
    predictions of shape ``(B,)``. Returns a ``(num_classes, num_classes)``
    integer tensor.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

    def update(self, preds: Tensor, targets: Tensor) -> None:
        """Accumulate a batch, applying argmax if preds are 2-D."""
        if preds.ndim == 2:
            preds = preds.argmax(dim=1)
        self._preds.append(preds)
        self._targets.append(targets)

    def compute(self) -> Tensor:
        """Return confusion matrix as an ``int64`` tensor via sklearn."""
        self._check_not_empty()
        y_pred = torch.cat(self._preds).cpu().numpy()
        y_true = torch.cat(self._targets).cpu().numpy()
        labels = list(range(self.num_classes))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        return torch.tensor(cm, dtype=torch.int64)
