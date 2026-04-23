"""Single transformer block: norm -> attention -> residual -> norm -> MLP -> residual."""

from __future__ import annotations

import torch
from torch import nn

from simpleml.models.vit.attention import Attention
from simpleml.models.vit.mlp import MLP


class Block(nn.Module):
    """Pre-norm transformer block.

    x = x + attn(norm1(x))
    x = x + mlp(norm2(x))

    Timm layout: ``norm1``, ``attn``, ``norm2``, ``mlp``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        # TODO: self.norm1, self.attn (Attention), self.norm2, self.mlp (MLP)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
