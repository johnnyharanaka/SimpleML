"""Tests for the ViT MLP block."""

import torch
from torch import nn

from simpleml.models.vit import MLP


class TestMLP:
    def test_is_nn_module(self):
        mlp = MLP(in_features=48, hidden_features=192)
        assert isinstance(mlp, nn.Module)

    def test_fc1_and_fc2_shapes(self):
        mlp = MLP(in_features=48, hidden_features=192, out_features=48)
        assert isinstance(mlp.fc1, nn.Linear)
        assert isinstance(mlp.fc2, nn.Linear)
        assert mlp.fc1.in_features == 48
        assert mlp.fc1.out_features == 192
        assert mlp.fc2.in_features == 192
        assert mlp.fc2.out_features == 48

    def test_act_is_gelu(self):
        mlp = MLP(in_features=48, hidden_features=192)
        assert isinstance(mlp.act, nn.GELU)

    def test_default_hidden_equals_in_features(self):
        mlp = MLP(in_features=48)
        assert mlp.fc1.out_features == 48
        assert mlp.fc2.in_features == 48

    def test_forward_output_shape(self):
        mlp = MLP(in_features=48, hidden_features=192)
        x = torch.randn(2, 17, 48)
        out = mlp(x)
        assert out.shape == (2, 17, 48)

    def test_state_dict_keys_match_timm(self):
        mlp = MLP(in_features=48, hidden_features=192)
        keys = set(mlp.state_dict().keys())
        assert keys == {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}
