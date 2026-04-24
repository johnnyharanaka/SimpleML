"""Tests for the simpleml.trainers module."""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from simpleml.configs import Config
from simpleml.metrics.accuracy import Accuracy
from simpleml.metrics.base import Metric
from simpleml.trainers.trainer import Trainer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_IN_FEATURES = 8
_NUM_CLASSES = 4
_NUM_SAMPLES = 64


def _make_dataset(n: int = _NUM_SAMPLES) -> TensorDataset:
    """Return a simple TensorDataset with random data."""
    x = torch.randn(n, _IN_FEATURES)
    y = torch.randint(0, _NUM_CLASSES, (n,))
    return TensorDataset(x, y)


def _make_model() -> nn.Module:
    """Return a small linear model."""
    return nn.Linear(_IN_FEATURES, _NUM_CLASSES)


def _make_trainer(
    val: bool = True,
    metrics: list[Metric] | None = None,
    scheduler: bool = False,
    training_config: dict | None = None,
) -> Trainer:
    """Build a Trainer with sensible defaults for testing."""
    model = _make_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = (
        torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        if scheduler
        else None
    )
    cfg = {"device": "cpu", "epochs": 2, "batch_size": 16, **(training_config or {})}
    return Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_dataset=_make_dataset(),
        val_dataset=_make_dataset(32) if val else None,
        scheduler=sched,
        metrics=metrics,
        training_config=cfg,
    )


# ---------------------------------------------------------------------------
# TestTrainerCreation
# ---------------------------------------------------------------------------


class TestTrainerCreation:
    def test_minimal_init(self) -> None:
        """Trainer can be created with only required args."""
        model = _make_model()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            train_dataset=_make_dataset(),
            training_config={"device": "cpu"},
        )
        assert trainer.val_loader is None
        assert trainer.metrics == []
        assert trainer.scheduler is None

    def test_full_init(self) -> None:
        """Trainer can be created with all optional args."""
        trainer = _make_trainer(val=True, metrics=[Accuracy()], scheduler=True)
        assert trainer.val_loader is not None
        assert len(trainer.metrics) == 1
        assert trainer.scheduler is not None

    def test_defaults_applied(self) -> None:
        """Training config defaults are applied when not overridden."""
        trainer = _make_trainer(training_config={"device": "cpu"})
        assert trainer._cfg["epochs"] == 2
        assert trainer._cfg["save_best"] is True
        assert trainer._cfg["num_workers"] == 0
        assert trainer._cfg["pin_memory"] is False

    def test_config_override(self) -> None:
        """User-provided config values override defaults."""
        trainer = _make_trainer(
            training_config={"device": "cpu", "batch_size": 8, "save_best": False}
        )
        assert trainer._cfg["batch_size"] == 8
        assert trainer._cfg["save_best"] is False

    def test_global_step_starts_at_zero(self) -> None:
        trainer = _make_trainer()
        assert trainer.global_step == 0

    def test_best_val_loss_starts_inf(self) -> None:
        trainer = _make_trainer()
        assert trainer.best_val_loss == math.inf


# ---------------------------------------------------------------------------
# TestResolveDevice
# ---------------------------------------------------------------------------


class TestResolveDevice:
    def test_cpu_explicit(self) -> None:
        device = Trainer._resolve_device("cpu")
        assert device == torch.device("cpu")

    def test_auto_fallback_cpu(self) -> None:
        with (
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            device = Trainer._resolve_device("auto")
            assert device == torch.device("cpu")

    def test_auto_prefers_mps(self) -> None:
        with (
            patch("torch.backends.mps.is_available", return_value=True),
            patch("torch.cuda.is_available", return_value=True),
        ):
            device = Trainer._resolve_device("auto")
            assert device == torch.device("mps")

    def test_auto_cuda_when_no_mps(self) -> None:
        with (
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=True),
        ):
            device = Trainer._resolve_device("auto")
            assert device == torch.device("cuda")

    def test_invalid_device_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown device"):
            Trainer._resolve_device("tpu")


# ---------------------------------------------------------------------------
# TestDataLoader
# ---------------------------------------------------------------------------


class TestDataLoader:
    def test_train_loader_created(self) -> None:
        trainer = _make_trainer()
        assert isinstance(trainer.train_loader, DataLoader)

    def test_val_loader_created(self) -> None:
        trainer = _make_trainer(val=True)
        assert isinstance(trainer.val_loader, DataLoader)

    def test_val_loader_none_when_no_val(self) -> None:
        trainer = _make_trainer(val=False)
        assert trainer.val_loader is None

    def test_batch_size(self) -> None:
        trainer = _make_trainer(training_config={"device": "cpu", "batch_size": 8})
        assert trainer.train_loader.batch_size == 8

    def test_train_loader_shuffles(self) -> None:
        trainer = _make_trainer()
        sampler = trainer.train_loader.sampler
        assert isinstance(sampler, torch.utils.data.sampler.RandomSampler)

    def test_val_loader_no_shuffle(self) -> None:
        trainer = _make_trainer(val=True)
        sampler = trainer.val_loader.sampler
        assert isinstance(sampler, torch.utils.data.sampler.SequentialSampler)


# ---------------------------------------------------------------------------
# TestTrainOneEpoch
# ---------------------------------------------------------------------------


class TestTrainOneEpoch:
    def test_returns_finite_loss(self) -> None:
        trainer = _make_trainer()
        loss = trainer._train_one_epoch(0)
        assert math.isfinite(loss)

    def test_global_step_increments(self) -> None:
        trainer = _make_trainer()
        assert trainer.global_step == 0
        trainer._train_one_epoch(0)
        assert trainer.global_step > 0

    def test_model_in_train_mode(self) -> None:
        trainer = _make_trainer()
        trainer.model.eval()
        trainer._train_one_epoch(0)
        assert trainer.model.training


# ---------------------------------------------------------------------------
# TestValidateOneEpoch
# ---------------------------------------------------------------------------


class TestValidateOneEpoch:
    def test_returns_loss(self) -> None:
        trainer = _make_trainer(val=True)
        result = trainer._validate_one_epoch(0)
        assert "loss" in result
        assert math.isfinite(result["loss"])

    def test_returns_metrics(self) -> None:
        trainer = _make_trainer(val=True, metrics=[Accuracy()])
        result = trainer._validate_one_epoch(0)
        assert "metrics" in result
        assert "Accuracy" in result["metrics"]

    def test_model_in_eval_mode(self) -> None:
        trainer = _make_trainer(val=True)
        trainer.model.train()
        trainer._validate_one_epoch(0)
        assert not trainer.model.training

    def test_empty_metrics(self) -> None:
        trainer = _make_trainer(val=True, metrics=[])
        result = trainer._validate_one_epoch(0)
        assert result["metrics"] == {}


# ---------------------------------------------------------------------------
# TestFit
# ---------------------------------------------------------------------------


class TestFit:
    def test_fit_returns_dict(self) -> None:
        trainer = _make_trainer()
        result = trainer.fit()
        expected_keys = {
            "last_train_loss",
            "last_val_loss",
            "best_val_loss",
            "last_metrics",
            "epochs_trained",
        }
        assert set(result.keys()) == expected_keys

    def test_fit_epochs_trained(self) -> None:
        trainer = _make_trainer(training_config={"device": "cpu", "epochs": 3})
        result = trainer.fit()
        assert result["epochs_trained"] == 3

    def test_fit_train_loss_finite(self) -> None:
        trainer = _make_trainer()
        result = trainer.fit()
        assert math.isfinite(result["last_train_loss"])

    def test_fit_val_loss_present(self) -> None:
        trainer = _make_trainer(val=True)
        result = trainer.fit()
        assert result["last_val_loss"] is not None
        assert math.isfinite(result["last_val_loss"])

    def test_fit_val_loss_none_without_val(self) -> None:
        trainer = _make_trainer(val=False)
        result = trainer.fit()
        assert result["last_val_loss"] is None

    def test_fit_best_val_loss_tracked(self) -> None:
        trainer = _make_trainer(val=True)
        result = trainer.fit()
        assert result["best_val_loss"] is not None
        assert math.isfinite(result["best_val_loss"])

    def test_fit_no_best_val_loss_without_val(self) -> None:
        trainer = _make_trainer(val=False)
        result = trainer.fit()
        assert result["best_val_loss"] is None

    def test_fit_metrics_returned(self) -> None:
        trainer = _make_trainer(val=True, metrics=[Accuracy()])
        result = trainer.fit()
        assert "Accuracy" in result["last_metrics"]

    def test_loss_decreases_on_trivial_data(self) -> None:
        """On easily separable data, loss should decrease over epochs."""
        torch.manual_seed(42)
        x = torch.randn(128, _IN_FEATURES)
        y = (x[:, 0] > 0).long()
        ds = TensorDataset(x, y)

        model = nn.Linear(_IN_FEATURES, 2)
        trainer = Trainer(
            model=model,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer=torch.optim.SGD(model.parameters(), lr=0.5),
            train_dataset=ds,
            training_config={"device": "cpu", "epochs": 20, "batch_size": 128},
        )
        first_loss = trainer._train_one_epoch(0)
        for i in range(1, 20):
            last_loss = trainer._train_one_epoch(i)
        assert last_loss < first_loss


# ---------------------------------------------------------------------------
# TestValidate
# ---------------------------------------------------------------------------


class TestValidate:
    def test_validate_standalone(self) -> None:
        trainer = _make_trainer(val=True)
        result = trainer.validate()
        assert "loss" in result
        assert "metrics" in result

    def test_validate_no_val_dataset_raises(self) -> None:
        trainer = _make_trainer(val=False)
        with pytest.raises(RuntimeError, match="No validation dataset"):
            trainer.validate()


# ---------------------------------------------------------------------------
# TestGradientClipping
# ---------------------------------------------------------------------------


class TestGradientClipping:
    def test_grad_clip_norm(self) -> None:
        trainer = _make_trainer(
            training_config={"device": "cpu", "grad_clip_norm": 1.0}
        )
        loss = trainer._train_one_epoch(0)
        assert math.isfinite(loss)

    def test_grad_clip_value(self) -> None:
        trainer = _make_trainer(
            training_config={"device": "cpu", "grad_clip_value": 0.5}
        )
        loss = trainer._train_one_epoch(0)
        assert math.isfinite(loss)

    def test_grad_clip_both(self) -> None:
        trainer = _make_trainer(
            training_config={
                "device": "cpu",
                "grad_clip_norm": 1.0,
                "grad_clip_value": 0.5,
            }
        )
        loss = trainer._train_one_epoch(0)
        assert math.isfinite(loss)


# ---------------------------------------------------------------------------
# TestCheckpointing
# ---------------------------------------------------------------------------


class TestCheckpointing:
    def test_save_and_load(self, tmp_path: Path) -> None:
        trainer = _make_trainer()
        trainer.global_step = 100
        trainer.best_val_loss = 0.5
        ckpt_path = tmp_path / "ckpt.pt"
        trainer.save_checkpoint(ckpt_path, epoch=3, val_loss=0.6)

        assert ckpt_path.exists()
        state = torch.load(ckpt_path, weights_only=False)
        assert state["epoch"] == 3
        assert state["global_step"] == 100
        assert state["val_loss"] == 0.6
        assert state["best_val_loss"] == 0.5

    def test_load_restores_state(self, tmp_path: Path) -> None:
        trainer = _make_trainer()
        trainer.global_step = 50
        trainer.best_val_loss = 0.3
        ckpt_path = tmp_path / "ckpt.pt"
        trainer.save_checkpoint(ckpt_path, epoch=5, val_loss=0.4)

        trainer2 = _make_trainer()
        resume_epoch = trainer2.load_checkpoint(ckpt_path)
        assert resume_epoch == 6
        assert trainer2.global_step == 50
        assert trainer2.best_val_loss == 0.3

    def test_save_creates_directory(self, tmp_path: Path) -> None:
        trainer = _make_trainer()
        ckpt_path = tmp_path / "sub" / "dir" / "ckpt.pt"
        trainer.save_checkpoint(ckpt_path, epoch=0)
        assert ckpt_path.exists()

    def test_save_best(self, tmp_path: Path) -> None:
        trainer = _make_trainer(
            val=True,
            training_config={
                "device": "cpu",
                "epochs": 2,
                "checkpoint_dir": str(tmp_path / "ckpts"),
                "save_best": True,
                "save_last": False,
                "save_every": None,
            },
        )
        trainer.fit()
        assert (tmp_path / "ckpts" / "best.pt").exists()

    def test_save_last(self, tmp_path: Path) -> None:
        trainer = _make_trainer(
            val=False,
            training_config={
                "device": "cpu",
                "epochs": 2,
                "checkpoint_dir": str(tmp_path / "ckpts"),
                "save_best": False,
                "save_last": True,
                "save_every": None,
            },
        )
        trainer.fit()
        assert (tmp_path / "ckpts" / "last.pt").exists()

    def test_save_every(self, tmp_path: Path) -> None:
        trainer = _make_trainer(
            val=False,
            training_config={
                "device": "cpu",
                "epochs": 4,
                "checkpoint_dir": str(tmp_path / "ckpts"),
                "save_best": False,
                "save_last": False,
                "save_every": 2,
            },
        )
        trainer.fit()
        assert (tmp_path / "ckpts" / "epoch_2.pt").exists()
        assert (tmp_path / "ckpts" / "epoch_4.pt").exists()
        assert not (tmp_path / "ckpts" / "epoch_1.pt").exists()

    def test_resume_from(self, tmp_path: Path) -> None:
        trainer1 = _make_trainer(
            val=False,
            training_config={
                "device": "cpu",
                "epochs": 2,
                "checkpoint_dir": str(tmp_path / "ckpts"),
                "save_last": True,
            },
        )
        trainer1.fit()

        ckpt_path = tmp_path / "ckpts" / "last.pt"
        trainer2 = _make_trainer(
            val=False,
            training_config={
                "device": "cpu",
                "epochs": 4,
                "resume_from": str(ckpt_path),
                "save_last": False,
                "save_best": False,
            },
        )
        result = trainer2.fit()
        assert result["epochs_trained"] == 2

    def test_checkpoint_with_scheduler(self, tmp_path: Path) -> None:
        trainer = _make_trainer(scheduler=True)
        ckpt_path = tmp_path / "ckpt.pt"
        trainer.save_checkpoint(ckpt_path, epoch=1)

        state = torch.load(ckpt_path, weights_only=False)
        assert state["scheduler_state_dict"] is not None

    def test_checkpoint_without_scheduler(self, tmp_path: Path) -> None:
        trainer = _make_trainer(scheduler=False)
        ckpt_path = tmp_path / "ckpt.pt"
        trainer.save_checkpoint(ckpt_path, epoch=1)

        state = torch.load(ckpt_path, weights_only=False)
        assert state["scheduler_state_dict"] is None


# ---------------------------------------------------------------------------
# TestTensorBoard
# ---------------------------------------------------------------------------


class TestTensorBoard:
    def test_log_dir_created(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "tb_logs"
        _make_trainer(training_config={"device": "cpu", "log_dir": str(log_dir)})
        assert log_dir.exists()

    def test_writer_closed_after_fit(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "tb_logs"
        trainer = _make_trainer(
            training_config={"device": "cpu", "log_dir": str(log_dir)}
        )
        trainer.fit()
        assert log_dir.exists()


# ---------------------------------------------------------------------------
# TestScheduler
# ---------------------------------------------------------------------------


class TestScheduler:
    def test_scheduler_step_per_epoch(self) -> None:
        trainer = _make_trainer(
            scheduler=True,
            training_config={"device": "cpu", "scheduler_step_on": "epoch"},
        )
        initial_lr = trainer.optimizer.param_groups[0]["lr"]
        trainer.fit()
        final_lr = trainer.optimizer.param_groups[0]["lr"]
        assert final_lr < initial_lr

    def test_scheduler_step_per_step(self) -> None:
        model = _make_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
        trainer = Trainer(
            model=model,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer=optimizer,
            train_dataset=_make_dataset(),
            scheduler=scheduler,
            training_config={
                "device": "cpu",
                "epochs": 2,
                "batch_size": 16,
                "scheduler_step_on": "step",
            },
        )
        initial_lr = optimizer.param_groups[0]["lr"]
        trainer.fit()
        final_lr = optimizer.param_groups[0]["lr"]
        assert final_lr < initial_lr


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_dataset_smaller_than_batch_size(self) -> None:
        model = _make_model()
        small_ds = _make_dataset(n=4)
        trainer = Trainer(
            model=model,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
            train_dataset=small_ds,
            training_config={"device": "cpu", "epochs": 1, "batch_size": 32},
        )
        result = trainer.fit()
        assert math.isfinite(result["last_train_loss"])

    def test_val_every_skips_epochs(self) -> None:
        trainer = _make_trainer(
            val=True,
            training_config={"device": "cpu", "epochs": 3, "val_every": 2},
        )
        result = trainer.fit()
        assert result["last_val_loss"] is not None

    def test_no_metrics_in_result(self) -> None:
        trainer = _make_trainer(val=True, metrics=[])
        result = trainer.fit()
        assert result["last_metrics"] == {}

    def test_mixed_precision_disabled_on_cpu(self) -> None:
        trainer = _make_trainer(
            training_config={"device": "cpu", "mixed_precision": True}
        )
        assert trainer._use_amp is False
        assert trainer._scaler is None


# ---------------------------------------------------------------------------
# TestFromConfig
# ---------------------------------------------------------------------------


class TestFromConfig:
    def test_from_config_yaml(self, tmp_path: Path) -> None:
        import yaml

        cfg_data = {
            "model": {
                "name": "ResNet",
                "params": {"variant": "resnet18", "num_classes": 4},
            },
            "dataset": {
                "train": {
                    "name": "ImageFolderDataset",
                    "params": {"root": str(tmp_path / "train")},
                },
            },
            "loss": {"name": "CrossEntropyLoss"},
            "optimizer": {"name": "SGD", "params": {"lr": 0.01}},
            "training": {"epochs": 1, "device": "cpu"},
        }

        train_dir = tmp_path / "train"
        for cls_name in ["cat", "dog"]:
            cls_dir = train_dir / cls_name
            cls_dir.mkdir(parents=True)
            from PIL import Image

            img = Image.new("RGB", (32, 32), color="red")
            img.save(cls_dir / "img1.jpg")

        yaml_path = tmp_path / "config.yaml"
        with open(yaml_path, "w") as f:
            yaml.safe_dump(cfg_data, f)

        trainer = Trainer.from_config(yaml_path)
        assert isinstance(trainer, Trainer)

    def test_from_config_object(self) -> None:
        cfg_data = {
            "model": {
                "name": "ResNet",
                "params": {"variant": "resnet18", "num_classes": _NUM_CLASSES},
            },
            "dataset": {
                "train": {
                    "name": "ImageFolderDataset",
                    "params": {"root": "/tmp/nonexistent"},
                },
            },
            "loss": {"name": "CrossEntropyLoss"},
            "optimizer": {"name": "Adam", "params": {"lr": 1e-3}},
            "training": {"epochs": 1, "device": "cpu"},
        }
        cfg = Config.from_dict(cfg_data)
        with pytest.raises(FileNotFoundError):
            Trainer.from_config(cfg)

    def test_from_config_dict(self) -> None:
        cfg_data = {
            "model": {
                "name": "ResNet",
                "params": {"variant": "resnet18", "num_classes": _NUM_CLASSES},
            },
            "dataset": {
                "train": {
                    "name": "ImageFolderDataset",
                    "params": {"root": "/tmp/nonexistent"},
                },
            },
            "loss": {"name": "CrossEntropyLoss"},
            "optimizer": {"name": "Adam", "params": {"lr": 1e-3}},
            "training": {"epochs": 1, "device": "cpu"},
        }
        with pytest.raises((FileNotFoundError, TypeError)):
            Trainer.from_config(cfg_data)
