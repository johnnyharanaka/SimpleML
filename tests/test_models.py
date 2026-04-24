"""Tests for the models module."""

import torch
from torch import nn

from simpleml import MODELS
from simpleml.models import ResNet


class TestResNet:
    def test_registered_in_models_registry(self):
        assert "ResNet" in MODELS

    def test_build_via_registry(self):
        model = MODELS.build("ResNet", variant="resnet18", num_classes=10)
        assert isinstance(model, ResNet)

    def test_forward_output_shape(self):
        model = ResNet(variant="resnet18", num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)

    def test_forward_resnet34(self):
        model = ResNet(variant="resnet34", num_classes=5)
        x = torch.randn(1, 3, 64, 64)
        out = model(x)
        assert out.shape == (1, 5)

    def test_forward_custom_in_chans(self):
        model = ResNet(variant="resnet18", num_classes=10, in_chans=1)
        x = torch.randn(2, 1, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)

    def test_num_classes_property(self):
        model = ResNet(variant="resnet18", num_classes=42)
        assert model.num_classes == 42

    def test_num_features_property(self):
        model = ResNet(variant="resnet18")
        assert isinstance(model.num_features, int)
        assert model.num_features > 0

    def test_is_nn_module(self):
        model = ResNet(variant="resnet18")
        assert isinstance(model, nn.Module)

    def test_default_variant(self):
        model = ResNet(num_classes=10)
        assert isinstance(model, nn.Module)
