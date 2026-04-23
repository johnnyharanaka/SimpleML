"""Tests for the ViT transformer Block."""

import torch
from torch import nn

from simpleml.models.vit import MLP, Attention, Block


class TestBlock:
    def test_is_nn_module(self):
        block = Block(embed_dim=48, num_heads=4)
        assert isinstance(block, nn.Module)

    def test_has_norm1_norm2_layernorm(self):
        block = Block(embed_dim=48, num_heads=4)
        assert isinstance(block.norm1, nn.LayerNorm)
        assert isinstance(block.norm2, nn.LayerNorm)
        assert block.norm1.normalized_shape == (48,)
        assert block.norm2.normalized_shape == (48,)

    def test_has_attention(self):
        block = Block(embed_dim=48, num_heads=4)
        assert isinstance(block.attn, Attention)

    def test_has_mlp_with_correct_hidden_dim(self):
        block = Block(embed_dim=48, num_heads=4, mlp_ratio=4.0)
        assert isinstance(block.mlp, MLP)
        assert block.mlp.fc1.out_features == 48 * 4

    def test_forward_output_shape(self):
        block = Block(embed_dim=48, num_heads=4)
        x = torch.randn(2, 17, 48)
        out = block(x)
        assert out.shape == (2, 17, 48)

    def test_forward_is_residual(self):
        """With zeroed attn and mlp outputs, forward should return x unchanged."""
        block = Block(embed_dim=48, num_heads=4)
        # Zero the output projections so attn(x) == 0 and mlp(x) == 0
        with torch.no_grad():
            block.attn.proj.weight.zero_()
            block.attn.proj.bias.zero_()
            block.mlp.fc2.weight.zero_()
            block.mlp.fc2.bias.zero_()
        x = torch.randn(2, 17, 48)
        out = block(x)
        assert torch.allclose(out, x, atol=1e-6)

    def test_state_dict_keys_match_timm(self):
        block = Block(embed_dim=48, num_heads=4)
        keys = set(block.state_dict().keys())
        expected = {
            "norm1.weight",
            "norm1.bias",
            "attn.qkv.weight",
            "attn.qkv.bias",
            "attn.proj.weight",
            "attn.proj.bias",
            "norm2.weight",
            "norm2.bias",
            "mlp.fc1.weight",
            "mlp.fc1.bias",
            "mlp.fc2.weight",
            "mlp.fc2.bias",
        }
        assert keys == expected
