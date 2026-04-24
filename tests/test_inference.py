"""Tests for the simpleml.inference module."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import TensorDataset

from simpleml.inference.predictor import PredictionResult, Predictor
from simpleml.metrics.accuracy import Accuracy
from simpleml.metrics.f1_score import F1Score

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_IN_FEATURES = 8
_NUM_CLASSES = 4
_NUM_SAMPLES = 32


def _make_model() -> nn.Module:
    """Return a small linear model."""
    return nn.Linear(_IN_FEATURES, _NUM_CLASSES)


def _make_dataset(n: int = _NUM_SAMPLES) -> TensorDataset:
    """Return a simple TensorDataset with random data."""
    x = torch.randn(n, _IN_FEATURES)
    y = torch.randint(0, _NUM_CLASSES, (n,))
    return TensorDataset(x, y)


def _make_predictor(
    metrics: list | None = None,
    inference_config: dict | None = None,
) -> Predictor:
    """Build a Predictor with sensible defaults for testing."""
    model = _make_model()
    cfg = {"device": "cpu", **(inference_config or {})}
    return Predictor(
        model=model,
        metrics=metrics,
        inference_config=cfg,
    )


def _create_image_dir(tmp_path: Path, n: int = 3) -> Path:
    """Create a directory with dummy images and return its path."""
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = Image.new("RGB", (8, 8), color=(i * 50, 100, 150))
        img.save(img_dir / f"img_{i}.jpg")
    return img_dir


def _create_image_folder_dataset(tmp_path: Path) -> Path:
    """Create an ImageFolderDataset-compatible directory tree."""
    root = tmp_path / "dataset"
    for cls_name in ["cat", "dog"]:
        cls_dir = root / cls_name
        cls_dir.mkdir(parents=True)
        for i in range(3):
            img = Image.new("RGB", (8, 8), color="red" if cls_name == "cat" else "blue")
            img.save(cls_dir / f"img_{i}.jpg")
    return root


# ---------------------------------------------------------------------------
# TestPredictorCreation
# ---------------------------------------------------------------------------


class TestPredictorCreation:
    def test_minimal_init(self) -> None:
        """Predictor can be created with only a model."""
        model = _make_model()
        predictor = Predictor(model=model, inference_config={"device": "cpu"})
        assert predictor.metrics == []
        assert predictor.transform is None

    def test_model_eval_mode(self) -> None:
        """Model is set to eval mode on init."""
        model = _make_model()
        model.train()
        predictor = Predictor(model=model, inference_config={"device": "cpu"})
        assert not predictor.model.training

    def test_with_metrics(self) -> None:
        """Predictor accepts metrics."""
        predictor = _make_predictor(metrics=[Accuracy()])
        assert len(predictor.metrics) == 1

    def test_device_config(self) -> None:
        """Device is resolved from config."""
        predictor = _make_predictor(inference_config={"device": "cpu"})
        assert predictor.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# TestFromConfig
# ---------------------------------------------------------------------------


class TestFromConfig:
    def test_from_config_dict(self) -> None:
        """Predictor can be built from a plain dict."""
        cfg = {
            "model": {
                "name": "ResNet",
                "params": {"variant": "resnet18", "num_classes": _NUM_CLASSES},
            },
            "inference": {"device": "cpu"},
        }
        predictor = Predictor.from_config(cfg)
        assert isinstance(predictor, Predictor)
        assert not predictor.model.training

    def test_from_config_yaml(self, tmp_path: Path) -> None:
        """Predictor can be built from a YAML file."""
        cfg = {
            "model": {
                "name": "ResNet",
                "params": {"variant": "resnet18", "num_classes": _NUM_CLASSES},
            },
            "inference": {"device": "cpu"},
        }
        yaml_path = tmp_path / "config.yaml"
        with open(yaml_path, "w") as f:
            yaml.safe_dump(cfg, f)

        predictor = Predictor.from_config(yaml_path)
        assert isinstance(predictor, Predictor)

    def test_from_config_with_metrics(self) -> None:
        """Metrics section is built from config."""
        cfg = {
            "model": {
                "name": "ResNet",
                "params": {"variant": "resnet18", "num_classes": _NUM_CLASSES},
            },
            "metrics": [
                {"name": "Accuracy"},
                {"name": "F1Score", "params": {"average": "weighted"}},
            ],
            "inference": {"device": "cpu"},
        }
        predictor = Predictor.from_config(cfg)
        assert len(predictor.metrics) == 2

    def test_from_config_stores_raw_config(self) -> None:
        """Raw config is stored for evaluate_from_config."""
        cfg = {
            "model": {
                "name": "ResNet",
                "params": {"variant": "resnet18", "num_classes": _NUM_CLASSES},
            },
            "inference": {"device": "cpu"},
        }
        predictor = Predictor.from_config(cfg)
        assert predictor._raw_config is not None

    def test_from_config_no_model_raises(self) -> None:
        """Missing model section raises ValueError."""
        with pytest.raises(ValueError, match="must contain a 'model' section"):
            Predictor.from_config({"inference": {"device": "cpu"}})

    def test_from_config_file_not_found(self) -> None:
        """Non-existent YAML path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            Predictor.from_config("/nonexistent/config.yaml")

    def test_from_config_trainer_checkpoint(self, tmp_path: Path) -> None:
        """Loads weights from a Trainer-format checkpoint (model_state_dict key)."""
        from simpleml.registries import MODELS

        ref_model = MODELS.build("ResNet", variant="resnet18", num_classes=_NUM_CLASSES)
        ckpt_path = tmp_path / "trainer_ckpt.pt"
        torch.save(
            {
                "epoch": 5,
                "model_state_dict": ref_model.state_dict(),
                "optimizer_state_dict": {},
            },
            ckpt_path,
        )

        cfg = {
            "model": {
                "name": "ResNet",
                "params": {"variant": "resnet18", "num_classes": _NUM_CLASSES},
            },
            "inference": {"device": "cpu"},
        }
        predictor = Predictor.from_config(cfg, checkpoint=ckpt_path)
        assert isinstance(predictor, Predictor)

    def test_from_config_pure_state_dict(self, tmp_path: Path) -> None:
        """Loads weights from a plain state_dict checkpoint."""
        cfg = {
            "model": {
                "name": "ResNet",
                "params": {"variant": "resnet18", "num_classes": _NUM_CLASSES},
            },
            "inference": {"device": "cpu"},
        }
        predictor_ref = Predictor.from_config(cfg)
        state_dict = predictor_ref.model.state_dict()

        ckpt_path = tmp_path / "state_dict.pt"
        torch.save(state_dict, ckpt_path)

        predictor = Predictor.from_config(cfg, checkpoint=ckpt_path)
        assert isinstance(predictor, Predictor)


# ---------------------------------------------------------------------------
# TestPredictImage
# ---------------------------------------------------------------------------


class TestPredictImage:
    def test_predict_pil_image(self) -> None:
        """Predict from a PIL Image."""
        model = nn.Linear(3 * 8 * 8, _NUM_CLASSES)

        class FlatModel(nn.Module):
            def __init__(self, linear: nn.Module) -> None:
                super().__init__()
                self.linear = linear

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x.reshape(x.size(0), -1))

        predictor = Predictor(
            model=FlatModel(model),
            inference_config={"device": "cpu"},
        )
        img = Image.new("RGB", (8, 8), color="red")
        result = predictor.predict_image(img)
        assert isinstance(result, PredictionResult)
        assert result.logits.shape == (1, _NUM_CLASSES)
        assert result.probabilities.shape == (1, _NUM_CLASSES)
        assert result.predicted_classes.shape == (1,)

    def test_predict_image_path(self, tmp_path: Path) -> None:
        """Predict from a file path."""

        class FlatModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(3 * 8 * 8, _NUM_CLASSES)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x.reshape(x.size(0), -1))

        predictor = Predictor(
            model=FlatModel(),
            inference_config={"device": "cpu"},
        )
        img = Image.new("RGB", (8, 8), color="green")
        img_path = tmp_path / "test.jpg"
        img.save(img_path)

        result = predictor.predict_image(img_path)
        assert isinstance(result, PredictionResult)
        assert result.logits.shape == (1, _NUM_CLASSES)

    def test_probabilities_sum_to_one(self) -> None:
        """Softmax probabilities should sum to ~1.0."""

        class FlatModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(3 * 8 * 8, _NUM_CLASSES)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x.reshape(x.size(0), -1))

        predictor = Predictor(
            model=FlatModel(),
            inference_config={"device": "cpu"},
        )
        img = Image.new("RGB", (8, 8), color="blue")
        result = predictor.predict_image(img)
        prob_sum = result.probabilities.sum(dim=1)
        assert torch.allclose(prob_sum, torch.ones(1), atol=1e-5)

    def test_predicted_class_in_range(self) -> None:
        """Predicted class should be within [0, num_classes)."""

        class FlatModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(3 * 8 * 8, _NUM_CLASSES)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x.reshape(x.size(0), -1))

        predictor = Predictor(
            model=FlatModel(),
            inference_config={"device": "cpu"},
        )
        img = Image.new("RGB", (8, 8), color="blue")
        result = predictor.predict_image(img)
        assert 0 <= result.predicted_classes.item() < _NUM_CLASSES


# ---------------------------------------------------------------------------
# TestPredictBatch
# ---------------------------------------------------------------------------


class TestPredictBatch:
    def _make_flat_predictor(self) -> Predictor:
        class FlatModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(3 * 8 * 8, _NUM_CLASSES)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x.reshape(x.size(0), -1))

        return Predictor(
            model=FlatModel(),
            inference_config={"device": "cpu", "batch_size": 2},
        )

    def test_predict_batch_list_of_paths(self, tmp_path: Path) -> None:
        """Predict from a list of image paths."""
        img_dir = _create_image_dir(tmp_path, n=4)
        paths = sorted(img_dir.glob("*.jpg"))
        predictor = self._make_flat_predictor()
        result = predictor.predict_batch(paths)
        assert result.logits.shape == (4, _NUM_CLASSES)
        assert result.probabilities.shape == (4, _NUM_CLASSES)
        assert result.predicted_classes.shape == (4,)

    def test_predict_batch_directory(self, tmp_path: Path) -> None:
        """Predict from a directory path."""
        img_dir = _create_image_dir(tmp_path, n=3)
        predictor = self._make_flat_predictor()
        result = predictor.predict_batch(img_dir)
        assert result.logits.shape == (3, _NUM_CLASSES)

    def test_predict_batch_probabilities_sum(self, tmp_path: Path) -> None:
        """All probability vectors should sum to ~1.0."""
        img_dir = _create_image_dir(tmp_path, n=3)
        predictor = self._make_flat_predictor()
        result = predictor.predict_batch(img_dir)
        sums = result.probabilities.sum(dim=1)
        assert torch.allclose(sums, torch.ones(3), atol=1e-5)


# ---------------------------------------------------------------------------
# TestPredictDataset
# ---------------------------------------------------------------------------


class TestPredictDataset:
    def test_predict_dataset(self) -> None:
        """Predict over a TensorDataset."""
        predictor = _make_predictor()
        ds = _make_dataset(16)
        result = predictor.predict_dataset(ds)
        assert result.logits.shape == (16, _NUM_CLASSES)
        assert result.probabilities.shape == (16, _NUM_CLASSES)
        assert result.predicted_classes.shape == (16,)

    def test_predict_dataset_probabilities_sum(self) -> None:
        """All probability vectors should sum to ~1.0."""
        predictor = _make_predictor()
        ds = _make_dataset(16)
        result = predictor.predict_dataset(ds)
        sums = result.probabilities.sum(dim=1)
        assert torch.allclose(sums, torch.ones(16), atol=1e-5)


# ---------------------------------------------------------------------------
# TestEvaluate
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_evaluate_returns_predictions_and_metrics(self) -> None:
        """Evaluate returns both predictions and metrics."""
        predictor = _make_predictor(metrics=[Accuracy()])
        ds = _make_dataset(16)
        result = predictor.evaluate(ds)
        assert "predictions" in result
        assert "metrics" in result
        assert isinstance(result["predictions"], PredictionResult)
        assert "Accuracy" in result["metrics"]

    def test_evaluate_multiple_metrics(self) -> None:
        """Multiple metrics are computed."""
        predictor = _make_predictor(metrics=[Accuracy(), F1Score(average="weighted")])
        ds = _make_dataset(16)
        result = predictor.evaluate(ds)
        assert "Accuracy" in result["metrics"]
        assert "F1Score" in result["metrics"]

    def test_evaluate_no_metrics(self) -> None:
        """Evaluate with no metrics returns empty dict."""
        predictor = _make_predictor(metrics=[])
        ds = _make_dataset(16)
        result = predictor.evaluate(ds)
        assert result["metrics"] == {}

    def test_evaluate_accuracy_value(self) -> None:
        """Accuracy should be between 0 and 1."""
        predictor = _make_predictor(metrics=[Accuracy()])
        ds = _make_dataset(64)
        result = predictor.evaluate(ds)
        acc = result["metrics"]["Accuracy"]
        assert 0.0 <= acc <= 1.0


# ---------------------------------------------------------------------------
# TestEvaluateFromConfig
# ---------------------------------------------------------------------------


class TestEvaluateFromConfig:
    def test_evaluate_from_config_test_split(self, tmp_path: Path) -> None:
        """evaluate_from_config works with 'test' split."""
        root = _create_image_folder_dataset(tmp_path)
        cfg = {
            "model": {
                "name": "ResNet",
                "params": {"variant": "resnet18", "num_classes": 2},
            },
            "dataset": {
                "test": {
                    "name": "ImageFolderDataset",
                    "params": {"root": str(root)},
                },
            },
            "metrics": [{"name": "Accuracy"}],
            "inference": {"device": "cpu", "batch_size": 4},
        }
        predictor = Predictor.from_config(cfg)
        result = predictor.evaluate_from_config("test")
        assert "predictions" in result
        assert "Accuracy" in result["metrics"]

    def test_evaluate_from_config_val_split(self, tmp_path: Path) -> None:
        """evaluate_from_config works with 'val' split."""
        root = _create_image_folder_dataset(tmp_path)
        cfg = {
            "model": {
                "name": "ResNet",
                "params": {"variant": "resnet18", "num_classes": 2},
            },
            "dataset": {
                "val": {
                    "name": "ImageFolderDataset",
                    "params": {"root": str(root)},
                },
            },
            "metrics": [{"name": "Accuracy"}],
            "inference": {"device": "cpu", "batch_size": 4},
        }
        predictor = Predictor.from_config(cfg)
        result = predictor.evaluate_from_config("val")
        assert "predictions" in result

    def test_evaluate_from_config_no_config_raises(self) -> None:
        """Raises RuntimeError when not built via from_config."""
        predictor = _make_predictor(metrics=[Accuracy()])
        with pytest.raises(RuntimeError, match="No config stored"):
            predictor.evaluate_from_config("test")

    def test_evaluate_from_config_missing_split_raises(self) -> None:
        """Raises KeyError when split not found in config."""
        cfg = {
            "model": {
                "name": "ResNet",
                "params": {"variant": "resnet18", "num_classes": _NUM_CLASSES},
            },
            "inference": {"device": "cpu"},
        }
        predictor = Predictor.from_config(cfg)
        with pytest.raises(KeyError, match="not found in config"):
            predictor.evaluate_from_config("test")
