"""Multi-head self-attention block."""

from __future__ import annotations

import torch
from torch import nn


class Attention(nn.Module):
    """Multi-head self-attention.

    Input : (B, N, embed_dim)
    Output: (B, N, embed_dim)

    Timm layout (required for weight loading):
        - ``qkv``: Linear(embed_dim, embed_dim * 3)   — Q, K, V concatenated
        - ``proj``: Linear(embed_dim, embed_dim)
        - ``attn_drop``, ``proj_drop``: Dropout layers
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 12,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        # TODO: self.qkv, self.proj, self.attn_drop, self.proj_drop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
