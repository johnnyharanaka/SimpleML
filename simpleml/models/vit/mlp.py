"""Feed-forward MLP block used inside each transformer block."""

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    """Two-layer MLP with GELU activation.

    Timm layout: ``fc1`` -> ``act`` (GELU) -> ``drop`` -> ``fc2`` -> ``drop``.

    Input : (B, N, in_features)
    Output: (B, N, out_features)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features or in_features
        self.out_features = out_features or in_features
        # TODO: self.fc1, self.act (GELU), self.fc2, self.drop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
