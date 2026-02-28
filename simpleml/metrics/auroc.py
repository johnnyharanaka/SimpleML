"""AUROC metric registered for config-driven use."""

from __future__ import annotations

import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor

from simpleml.metrics.base import Metric
from simpleml.registries import METRICS


@METRICS.register
class AUROC(Metric):
    """Area Under the Receiver Operating Characteristic curve.

    Expects **probability** predictions (after softmax), *not* argmaxed
    class indices. For binary classification, preds may be shape ``(B,)``
    or ``(B, 2)``. For multiclass, preds must be shape ``(B, C)``.
    """

    def __init__(self, multi_class: str = "ovr") -> None:
        super().__init__()
        self.multi_class = multi_class

    def update(self, preds: Tensor, targets: Tensor) -> None:
        """Accumulate probability predictions (no argmax applied)."""
        self._preds.append(preds)
        self._targets.append(targets)

    def compute(self) -> float:
        """Return AUROC as a float via ``sklearn.metrics.roc_auc_score``."""
        self._check_not_empty()
        y_score = torch.cat(self._preds).cpu().numpy()
        y_true = torch.cat(self._targets).cpu().numpy()
        if y_score.ndim == 1 or y_score.shape[1] == 1:
            y_score = y_score.ravel()
            return float(roc_auc_score(y_true, y_score))
        return float(roc_auc_score(y_true, y_score, multi_class=self.multi_class))
