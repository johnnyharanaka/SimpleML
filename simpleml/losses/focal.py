"""Focal loss for handling class imbalance (Lin et al., 2017)."""

from __future__ import annotations

import torch
import torch.nn.functional as f
from torch import Tensor, nn

from simpleml.registries import LOSSES


@LOSSES.register
class FocalLoss(nn.Module):
    """Focal loss that down-weights well-classified examples.

    Applies a modulating factor ``(1 - p_t)^gamma`` to the standard
    cross-entropy loss so that hard examples receive higher weight.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute focal loss.

        Args:
            input: Logits of shape ``(B, C)``.
            target: Integer class labels of shape ``(B,)``.

        Returns:
            Scalar loss (or unreduced per-sample losses depending on *reduction*).
        """
        ce_loss = f.cross_entropy(input, target, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_weight = self.alpha * (1.0 - p_t) ** self.gamma
        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
