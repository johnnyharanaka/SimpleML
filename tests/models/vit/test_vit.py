"""Tests for the full ViT model."""

import pytest
import torch
from torch import nn

from simpleml import MODELS
from simpleml.models.vit import Block, PatchEmbed, ViT


class TestViT:
    def test_registered_in_models_registry(self):
        assert "ViT" in MODELS

    def test_build_via_registry(self):
        model = MODELS.build(
            "ViT",
            img_size=32,
            patch_size=8,
            embed_dim=48,
            depth=2,
            num_heads=4,
            num_classes=10,
        )
        assert isinstance(model, ViT)

    def test_is_nn_module(self):
        model = ViT(img_size=32, patch_size=8, embed_dim=48, depth=2, num_heads=4)
        assert isinstance(model, nn.Module)

    def test_has_patch_embed(self):
        model = ViT(img_size=32, patch_size=8, embed_dim=48, depth=2, num_heads=4)
        assert isinstance(model.patch_embed, PatchEmbed)

    def test_has_cls_token_shape(self):
        model = ViT(img_size=32, patch_size=8, embed_dim=48, depth=2, num_heads=4)
        assert isinstance(model.cls_token, nn.Parameter)
        assert model.cls_token.shape == (1, 1, 48)

    def test_has_pos_embed_shape(self):
        model = ViT(img_size=32, patch_size=8, embed_dim=48, depth=2, num_heads=4)
        # num_patches = (32 // 8) ** 2 = 16, +1 for cls
        assert isinstance(model.pos_embed, nn.Parameter)
        assert model.pos_embed.shape == (1, 17, 48)

    def test_has_blocks_of_correct_depth(self):
        model = ViT(img_size=32, patch_size=8, embed_dim=48, depth=3, num_heads=4)
        assert isinstance(model.blocks, nn.ModuleList)
        assert len(model.blocks) == 3
        for b in model.blocks:
            assert isinstance(b, Block)

    def test_has_final_norm(self):
        model = ViT(img_size=32, patch_size=8, embed_dim=48, depth=2, num_heads=4)
        assert isinstance(model.norm, nn.LayerNorm)
        assert model.norm.normalized_shape == (48,)

    def test_has_head(self):
        model = ViT(
            img_size=32, patch_size=8, embed_dim=48, depth=2, num_heads=4, num_classes=10
        )
        assert isinstance(model.head, nn.Linear)
        assert model.head.in_features == 48
        assert model.head.out_features == 10

    def test_forward_output_shape(self):
        model = ViT(
            img_size=32,
            patch_size=8,
            embed_dim=48,
            depth=2,
            num_heads=4,
            num_classes=10,
        )
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)

    def test_state_dict_keys_match_timm_vit_tiny(self):
        """Verify the parameter layout matches timm's vit_tiny_patch16_224,
        so that pretrained .pth weights can be loaded directly with load_state_dict."""
        timm = pytest.importorskip("timm")
        timm_model = timm.create_model("vit_tiny_patch16_224", pretrained=False)

        our_model = ViT(
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=1000,
            embed_dim=192,
            depth=12,
            num_heads=3,
            mlp_ratio=4.0,
            qkv_bias=True,
        )

        timm_keys = set(timm_model.state_dict().keys())
        our_keys = set(our_model.state_dict().keys())

        # Every key we declare must exist in timm (we can be a strict subset while
        # ignoring auxiliary timm-only keys like norm_pre, fc_norm, etc.).
        missing_in_timm = our_keys - timm_keys
        assert missing_in_timm == set(), f"Our model has keys timm doesn't: {missing_in_timm}"

        # Every core ViT key timm has must exist in our model.
        core_required = {
            "cls_token",
            "pos_embed",
            "patch_embed.proj.weight",
            "patch_embed.proj.bias",
            "norm.weight",
            "norm.bias",
            "head.weight",
            "head.bias",
        }
        for i in range(12):
            core_required.update(
                {
                    f"blocks.{i}.norm1.weight",
                    f"blocks.{i}.norm1.bias",
                    f"blocks.{i}.attn.qkv.weight",
                    f"blocks.{i}.attn.qkv.bias",
                    f"blocks.{i}.attn.proj.weight",
                    f"blocks.{i}.attn.proj.bias",
                    f"blocks.{i}.norm2.weight",
                    f"blocks.{i}.norm2.bias",
                    f"blocks.{i}.mlp.fc1.weight",
                    f"blocks.{i}.mlp.fc1.bias",
                    f"blocks.{i}.mlp.fc2.weight",
                    f"blocks.{i}.mlp.fc2.bias",
                }
            )
        missing = core_required - our_keys
        assert missing == set(), f"Missing required ViT keys: {missing}"

    def test_load_timm_weights_shapes_align(self):
        """Tensor shapes for each core key must match timm's so load_state_dict works."""
        timm = pytest.importorskip("timm")
        timm_model = timm.create_model("vit_tiny_patch16_224", pretrained=False)

        our_model = ViT(
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=1000,
            embed_dim=192,
            depth=12,
            num_heads=3,
            mlp_ratio=4.0,
            qkv_bias=True,
        )

        timm_sd = timm_model.state_dict()
        our_sd = our_model.state_dict()
        for key, tensor in our_sd.items():
            assert tensor.shape == timm_sd[key].shape, (
                f"Shape mismatch on '{key}': ours={tuple(tensor.shape)} "
                f"timm={tuple(timm_sd[key].shape)}"
            )
