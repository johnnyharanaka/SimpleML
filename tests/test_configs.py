"""Tests for the simpleml.configs module."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from simpleml.configs import Config
from simpleml.metrics.base import Metric
from simpleml.registries import LOSSES, MODELS, OPTIMIZERS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_config() -> dict:
    """Return a minimal valid config dict."""
    return {
        "model": {
            "name": "ResNet",
            "params": {"variant": "resnet18", "num_classes": 10},
        },
        "dataset": {
            "train": {"name": "ImageFolderDataset", "params": {"root": "/tmp/train"}},
        },
        "loss": {"name": "CrossEntropyLoss"},
        "optimizer": {"name": "Adam", "params": {"lr": 1e-3}},
    }


def _full_config() -> dict:
    """Return a config dict with all sections populated."""
    return {
        **_minimal_config(),
        "dataset": {
            "train": {"name": "ImageFolderDataset", "params": {"root": "/tmp/train"}},
            "val": {"name": "ImageFolderDataset", "params": {"root": "/tmp/val"}},
        },
        "scheduler": {"name": "StepLR", "params": {"step_size": 10}},
        "metrics": [
            {"name": "Accuracy"},
            {"name": "F1Score", "params": {"average": "weighted"}},
        ],
        "training": {"epochs": 10, "batch_size": 32, "device": "mps"},
    }


def _write_yaml(path, data: dict) -> None:
    """Write a dict to a YAML file."""
    import yaml

    with open(path, "w") as f:
        yaml.safe_dump(data, f)


# ---------------------------------------------------------------------------
# TestConfigCreation
# ---------------------------------------------------------------------------


class TestConfigCreation:
    def test_from_dict(self) -> None:
        cfg = Config.from_dict(_minimal_config())
        assert isinstance(cfg, Config)

    def test_from_yaml(self, tmp_path) -> None:
        path = tmp_path / "config.yaml"
        _write_yaml(path, _minimal_config())
        cfg = Config.from_yaml(path)
        assert isinstance(cfg, Config)

    def test_from_yaml_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            Config.from_yaml("/nonexistent/path/config.yaml")

    def test_from_yaml_invalid_content(self, tmp_path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("just a string")
        with pytest.raises(TypeError, match="YAML root must be a mapping"):
            Config.from_yaml(path)


# ---------------------------------------------------------------------------
# TestConfigValidation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_missing_required_section(self) -> None:
        data = _minimal_config()
        del data["model"]
        with pytest.raises(ValueError, match="Missing required"):
            Config(data)

    def test_missing_multiple_sections(self) -> None:
        with pytest.raises(ValueError, match="Missing required"):
            Config({"model": {"name": "X"}})

    def test_unknown_section(self) -> None:
        data = _minimal_config()
        data["typo_section"] = {"name": "X"}
        with pytest.raises(ValueError, match="Unknown config sections"):
            Config(data)

    def test_model_missing_name(self) -> None:
        data = _minimal_config()
        data["model"] = {"params": {"variant": "resnet18"}}
        with pytest.raises(ValueError, match="must contain a 'name' key"):
            Config(data)

    def test_model_not_a_dict(self) -> None:
        data = _minimal_config()
        data["model"] = "ResNet"
        with pytest.raises(TypeError, match="'model' section must be a mapping"):
            Config(data)

    def test_params_not_a_dict(self) -> None:
        data = _minimal_config()
        data["model"]["params"] = "invalid"
        with pytest.raises(TypeError, match="'model.params' must be a mapping"):
            Config(data)

    def test_dataset_missing_train(self) -> None:
        data = _minimal_config()
        data["dataset"] = {"val": {"name": "ImageFolderDataset"}}
        with pytest.raises(ValueError, match="must contain a 'train' split"):
            Config(data)

    def test_dataset_not_a_dict(self) -> None:
        data = _minimal_config()
        data["dataset"] = "ImageFolderDataset"
        with pytest.raises(TypeError, match="'dataset' section must be a mapping"):
            Config(data)

    def test_dataset_split_missing_name(self) -> None:
        data = _minimal_config()
        data["dataset"]["train"] = {"params": {"root": "/tmp"}}
        with pytest.raises(ValueError, match="must contain a 'name' key"):
            Config(data)

    def test_metrics_not_a_list(self) -> None:
        data = _minimal_config()
        data["metrics"] = {"name": "Accuracy"}
        with pytest.raises(TypeError, match="'metrics' section must be a list"):
            Config(data)

    def test_metrics_entry_missing_name(self) -> None:
        data = _minimal_config()
        data["metrics"] = [{"params": {"average": "macro"}}]
        with pytest.raises(ValueError, match="must contain a 'name' key"):
            Config(data)

    def test_metrics_entry_not_a_dict(self) -> None:
        data = _minimal_config()
        data["metrics"] = ["Accuracy"]
        with pytest.raises(TypeError, match="metrics\\[0\\] must be a mapping"):
            Config(data)

    def test_metrics_params_not_a_dict(self) -> None:
        data = _minimal_config()
        data["metrics"] = [{"name": "F1Score", "params": "weighted"}]
        with pytest.raises(TypeError, match="metrics\\[0\\].params must be a mapping"):
            Config(data)

    def test_model_weights_not_a_string(self) -> None:
        data = _minimal_config()
        data["model"]["weights"] = 123
        with pytest.raises(TypeError, match="'model.weights' must be a string"):
            Config(data)

    def test_scheduler_validated(self) -> None:
        data = _minimal_config()
        data["scheduler"] = {"missing_name": True}
        with pytest.raises(ValueError, match="must contain a 'name' key"):
            Config(data)

    def test_data_not_a_dict(self) -> None:
        with pytest.raises(TypeError, match="Config data must be a dict"):
            Config([1, 2, 3])

    def test_valid_full_config(self) -> None:
        cfg = Config(_full_config())
        assert isinstance(cfg, Config)


# ---------------------------------------------------------------------------
# TestConfigProperties
# ---------------------------------------------------------------------------


class TestConfigProperties:
    def test_data_returns_copy(self) -> None:
        original = _minimal_config()
        cfg = Config(original)
        data = cfg.data
        data["model"]["name"] = "MUTATED"
        assert cfg.data["model"]["name"] == "ResNet"

    def test_training_present(self) -> None:
        cfg = Config(_full_config())
        assert cfg.training["epochs"] == 10
        assert cfg.training["batch_size"] == 32

    def test_training_absent(self) -> None:
        cfg = Config(_minimal_config())
        assert cfg.training == {}

    def test_training_returns_copy(self) -> None:
        cfg = Config(_full_config())
        t = cfg.training
        t["epochs"] = 999
        assert cfg.training["epochs"] == 10

    def test_repr(self) -> None:
        cfg = Config(_minimal_config())
        r = repr(cfg)
        assert "Config" in r
        assert "dataset" in r
        assert "model" in r


# ---------------------------------------------------------------------------
# TestConfigBuildModel
# ---------------------------------------------------------------------------


class TestConfigBuildModel:
    def test_build_model(self) -> None:
        cfg = Config(_minimal_config())
        model = cfg.build_model()
        assert isinstance(model, nn.Module)

    def test_build_model_params(self) -> None:
        cfg = Config(_minimal_config())
        model = cfg.build_model()
        assert model.num_classes == 10

    def test_build_model_registry(self) -> None:
        cfg = Config(_minimal_config())
        model = cfg.build_model()
        assert type(model).__name__ in MODELS

    def test_build_model_with_weights(self, tmp_path) -> None:
        cfg = Config(_minimal_config())
        ref_model = cfg.build_model()
        weights_path = tmp_path / "weights.pt"
        torch.save(ref_model.state_dict(), weights_path)

        data = _minimal_config()
        data["model"]["weights"] = str(weights_path)
        cfg2 = Config(data)
        loaded = cfg2.build_model()
        for p1, p2 in zip(ref_model.parameters(), loaded.parameters()):
            assert torch.equal(p1, p2)

    def test_build_model_weights_file_not_found(self) -> None:
        data = _minimal_config()
        data["model"]["weights"] = "/nonexistent/weights.pt"
        cfg = Config(data)
        with pytest.raises(FileNotFoundError, match="Weights file not found"):
            cfg.build_model()

    def test_build_model_without_weights(self) -> None:
        cfg = Config(_minimal_config())
        model = cfg.build_model()
        assert isinstance(model, nn.Module)
        assert "weights" not in _minimal_config()["model"]


# ---------------------------------------------------------------------------
# TestConfigBuildLoss
# ---------------------------------------------------------------------------


class TestConfigBuildLoss:
    def test_build_loss(self) -> None:
        cfg = Config(_minimal_config())
        loss = cfg.build_loss()
        assert isinstance(loss, nn.Module)

    def test_build_loss_with_params(self) -> None:
        data = _minimal_config()
        data["loss"] = {"name": "FocalLoss", "params": {"gamma": 3.0}}
        cfg = Config(data)
        loss = cfg.build_loss()
        assert loss.gamma == 3.0

    def test_build_loss_registry(self) -> None:
        cfg = Config(_minimal_config())
        loss = cfg.build_loss()
        assert type(loss).__name__ in LOSSES


# ---------------------------------------------------------------------------
# TestConfigBuildDataset
# ---------------------------------------------------------------------------


class TestConfigBuildDataset:
    def test_build_dataset_missing_split(self) -> None:
        cfg = Config(_minimal_config())
        with pytest.raises(KeyError, match="val"):
            cfg.build_dataset("val")


# ---------------------------------------------------------------------------
# TestConfigBuildOptimizer
# ---------------------------------------------------------------------------


class TestConfigBuildOptimizer:
    def test_build_optimizer(self) -> None:
        cfg = Config(_minimal_config())
        model = cfg.build_model()
        optimizer = cfg.build_optimizer(model)
        assert isinstance(optimizer, torch.optim.Optimizer)

    def test_build_optimizer_lr(self) -> None:
        cfg = Config(_minimal_config())
        model = cfg.build_model()
        optimizer = cfg.build_optimizer(model)
        assert optimizer.defaults["lr"] == 1e-3

    def test_build_optimizer_registry(self) -> None:
        cfg = Config(_minimal_config())
        model = cfg.build_model()
        optimizer = cfg.build_optimizer(model)
        assert type(optimizer).__name__ in OPTIMIZERS


# ---------------------------------------------------------------------------
# TestConfigBuildScheduler
# ---------------------------------------------------------------------------


class TestConfigBuildScheduler:
    def test_build_scheduler_none(self) -> None:
        cfg = Config(_minimal_config())
        model = cfg.build_model()
        optimizer = cfg.build_optimizer(model)
        assert cfg.build_scheduler(optimizer) is None

    def test_build_scheduler(self) -> None:
        cfg = Config(_full_config())
        model = cfg.build_model()
        optimizer = cfg.build_optimizer(model)
        scheduler = cfg.build_scheduler(optimizer)
        assert scheduler is not None
        assert isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler)


# ---------------------------------------------------------------------------
# TestConfigBuildMetrics
# ---------------------------------------------------------------------------


class TestConfigBuildMetrics:
    def test_build_metrics_none(self) -> None:
        cfg = Config(_minimal_config())
        assert cfg.build_metrics() is None

    def test_build_metrics(self) -> None:
        cfg = Config(_full_config())
        metrics = cfg.build_metrics()
        assert isinstance(metrics, list)
        assert len(metrics) == 2

    def test_build_metrics_types(self) -> None:
        cfg = Config(_full_config())
        metrics = cfg.build_metrics()
        for m in metrics:
            assert isinstance(m, Metric)

    def test_build_metrics_params(self) -> None:
        cfg = Config(_full_config())
        metrics = cfg.build_metrics()
        f1 = metrics[1]
        assert f1.average == "weighted"


# ---------------------------------------------------------------------------
# TestConfigBuildAll
# ---------------------------------------------------------------------------


class TestConfigBuildAll:
    def test_build_all_minimal(self) -> None:
        cfg = Config(_minimal_config())
        components = cfg.build_all()
        assert isinstance(components["model"], nn.Module)
        assert isinstance(components["loss"], nn.Module)
        assert isinstance(components["optimizer"], torch.optim.Optimizer)
        assert components["scheduler"] is None
        assert components["metrics"] is None

    def test_build_all_full(self) -> None:
        cfg = Config(_full_config())
        components = cfg.build_all()
        assert isinstance(components["model"], nn.Module)
        assert isinstance(components["loss"], nn.Module)
        assert isinstance(components["optimizer"], torch.optim.Optimizer)
        assert components["scheduler"] is not None
        assert isinstance(components["metrics"], list)
        assert len(components["metrics"]) == 2

    def test_build_all_with_external_model(self) -> None:
        cfg = Config(_minimal_config())
        model = cfg.build_model()
        components = cfg.build_all(model=model)
        assert components["model"] is model

    def test_build_all_keys(self) -> None:
        cfg = Config(_minimal_config())
        components = cfg.build_all()
        expected = {"model", "loss", "optimizer", "scheduler", "metrics"}
        assert set(components.keys()) == expected
