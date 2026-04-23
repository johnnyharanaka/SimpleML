"""Patch embedding: splits an image into patches and projects each to an embedding vector."""

from __future__ import annotations

import torch
from torch import nn


class PatchEmbed(nn.Module):
    """Split an image into non-overlapping patches and project each to an ``embed_dim`` vector.

    Input : (B, in_chans, img_size, img_size)
    Output: (B, num_patches, embed_dim)   where num_patches = (img_size // patch_size) ** 2

    Note: timm implements this with a Conv2d whose ``kernel=stride=patch_size`` (equivalent
    to applying the same linear projection to every patch). The submodule is named ``proj``
    to keep ``state_dict`` compatibility with timm's ViT.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.proj: nn.Conv2d  # define the Conv2d in your implementation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
