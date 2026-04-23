"""Tests for the ViT Attention block."""

import pytest
import torch
from torch import nn

from simpleml.models.vit import Attention


class TestAttention:
    def test_is_nn_module(self):
        attn = Attention(embed_dim=48, num_heads=4)
        assert isinstance(attn, nn.Module)

    def test_embed_dim_divisible_by_num_heads(self):
        with pytest.raises(AssertionError):
            Attention(embed_dim=50, num_heads=4)

    def test_head_dim_and_scale(self):
        attn = Attention(embed_dim=48, num_heads=4)
        assert attn.head_dim == 12
        assert attn.scale == pytest.approx(12**-0.5)

    def test_qkv_layer(self):
        attn = Attention(embed_dim=48, num_heads=4, qkv_bias=True)
        assert isinstance(attn.qkv, nn.Linear)
        assert attn.qkv.in_features == 48
        assert attn.qkv.out_features == 48 * 3
        assert attn.qkv.bias is not None

    def test_qkv_bias_false(self):
        attn = Attention(embed_dim=48, num_heads=4, qkv_bias=False)
        assert attn.qkv.bias is None

    def test_proj_layer(self):
        attn = Attention(embed_dim=48, num_heads=4)
        assert isinstance(attn.proj, nn.Linear)
        assert attn.proj.in_features == 48
        assert attn.proj.out_features == 48

    def test_forward_output_shape(self):
        attn = Attention(embed_dim=48, num_heads=4)
        x = torch.randn(2, 17, 48)
        out = attn(x)
        assert out.shape == (2, 17, 48)

    def test_state_dict_keys_match_timm(self):
        attn = Attention(embed_dim=48, num_heads=4, qkv_bias=True)
        keys = set(attn.state_dict().keys())
        assert keys == {"qkv.weight", "qkv.bias", "proj.weight", "proj.bias"}
