"""Tests for the ViT PatchEmbed block."""

import torch
from torch import nn

from simpleml.models.vit import PatchEmbed


class TestPatchEmbed:
    def test_is_nn_module(self):
        pe = PatchEmbed(img_size=224, patch_size=16, in_chans=3, embed_dim=768)
        assert isinstance(pe, nn.Module)

    def test_num_patches(self):
        pe = PatchEmbed(img_size=224, patch_size=16)
        assert pe.num_patches == (224 // 16) ** 2  # 196

    def test_num_patches_small(self):
        pe = PatchEmbed(img_size=32, patch_size=8)
        assert pe.num_patches == (32 // 8) ** 2  # 16

    def test_proj_is_conv2d(self):
        pe = PatchEmbed(img_size=224, patch_size=16, in_chans=3, embed_dim=768)
        assert isinstance(pe.proj, nn.Conv2d)
        assert pe.proj.in_channels == 3
        assert pe.proj.out_channels == 768
        assert pe.proj.kernel_size == (16, 16)
        assert pe.proj.stride == (16, 16)

    def test_forward_output_shape(self):
        pe = PatchEmbed(img_size=32, patch_size=8, in_chans=3, embed_dim=48)
        x = torch.randn(2, 3, 32, 32)
        out = pe(x)
        assert out.shape == (2, 16, 48)  # (B, num_patches, embed_dim)

    def test_forward_default_shape(self):
        pe = PatchEmbed(img_size=224, patch_size=16, in_chans=3, embed_dim=768)
        x = torch.randn(1, 3, 224, 224)
        out = pe(x)
        assert out.shape == (1, 196, 768)

    def test_state_dict_keys_match_timm(self):
        pe = PatchEmbed(img_size=224, patch_size=16, in_chans=3, embed_dim=768)
        keys = set(pe.state_dict().keys())
        assert keys == {"proj.weight", "proj.bias"}
