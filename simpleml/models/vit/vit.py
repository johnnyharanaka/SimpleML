"""Vision Transformer (ViT) — Dosovitskiy et al., 2020."""

from __future__ import annotations

import torch
from torch import nn

from simpleml.models.vit.block import Block
from simpleml.models.vit.patch_embed import PatchEmbed
from simpleml.registries import MODELS


@MODELS.register
class ViT(nn.Module):
    """Vision Transformer.

    Parameter names follow timm's ViT layout so that pretrained weights can be loaded
    directly with ``load_state_dict``. Expected top-level submodules / parameters:

        - ``patch_embed`` (PatchEmbed, with inner ``proj`` Conv2d)
        - ``cls_token``   (nn.Parameter, shape (1, 1, embed_dim))
        - ``pos_embed``   (nn.Parameter, shape (1, num_patches + 1, embed_dim))
        - ``blocks``      (nn.ModuleList of ``Block``, length = depth)
        - ``norm``        (LayerNorm(embed_dim))
        - ``head``        (Linear(embed_dim, num_classes))

    Forward pass:
        1. patches = patch_embed(x)                            # (B, N, D)
        2. prepend cls_token                                   # (B, N+1, D)
        3. add pos_embed
        4. pass through each transformer block
        5. apply final norm
        6. classify from the cls token -> head                 # (B, num_classes)
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        # TODO: patch_embed, cls_token, pos_embed, blocks, norm, head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
