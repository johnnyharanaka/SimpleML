"""Tests for the simpleml.metrics module."""

import pytest
import torch

from simpleml.metrics.base import Metric
from simpleml.registries import METRICS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

B, C = 32, 4


def _logits_and_targets() -> tuple[torch.Tensor, torch.Tensor]:
    """Return classification logits (B, C) and integer targets (B,)."""
    return torch.randn(B, C), torch.randint(0, C, (B,))


def _perfect_preds_and_targets() -> tuple[torch.Tensor, torch.Tensor]:
    """Return one-hot logits that perfectly predict the targets."""
    targets = torch.randint(0, C, (B,))
    preds = torch.zeros(B, C)
    preds[torch.arange(B), targets] = 10.0
    return preds, targets


def _binary_probs_and_targets() -> tuple[torch.Tensor, torch.Tensor]:
    """Return binary probability scores (B,) and binary targets (B,)."""
    targets = torch.randint(0, 2, (B,))
    probs = targets.float() * 0.8 + (1 - targets.float()) * 0.2
    probs += torch.randn(B) * 0.05
    return probs.clamp(0.0, 1.0), targets


def _multiclass_probs_and_targets() -> tuple[torch.Tensor, torch.Tensor]:
    """Return softmax probabilities (B, C) and integer targets (B,)."""
    logits, targets = _logits_and_targets()
    probs = torch.softmax(logits, dim=1)
    return probs, targets


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class TestMetricBase:
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            Metric()  # type: ignore[abstract]

    def test_check_not_empty_raises(self) -> None:
        metric = METRICS.build("Accuracy")
        with pytest.raises(RuntimeError, match="called before any update"):
            metric.compute()


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------


class TestAccuracy:
    def test_registered(self) -> None:
        assert "Accuracy" in METRICS

    def test_build(self) -> None:
        metric = METRICS.build("Accuracy")
        assert isinstance(metric, Metric)

    def test_perfect_accuracy(self) -> None:
        preds, targets = _perfect_preds_and_targets()
        metric = METRICS.build("Accuracy")
        result = metric(preds, targets)
        assert result == pytest.approx(1.0)

    def test_with_logits(self) -> None:
        logits, targets = _logits_and_targets()
        metric = METRICS.build("Accuracy")
        result = metric(logits, targets)
        assert 0.0 <= result <= 1.0

    def test_with_1d_preds(self) -> None:
        targets = torch.randint(0, C, (B,))
        preds = targets.clone()
        metric = METRICS.build("Accuracy")
        result = metric(preds, targets)
        assert result == pytest.approx(1.0)

    def test_multi_batch_accumulation(self) -> None:
        metric = METRICS.build("Accuracy")
        preds1, targets1 = _perfect_preds_and_targets()
        preds2, targets2 = _perfect_preds_and_targets()
        metric.update(preds1, targets1)
        metric.update(preds2, targets2)
        result = metric.compute()
        assert result == pytest.approx(1.0)

    def test_reset_clears_state(self) -> None:
        metric = METRICS.build("Accuracy")
        preds, targets = _logits_and_targets()
        metric.update(preds, targets)
        metric.reset()
        with pytest.raises(RuntimeError):
            metric.compute()


# ---------------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------------


class TestPrecision:
    def test_registered(self) -> None:
        assert "Precision" in METRICS

    def test_build(self) -> None:
        metric = METRICS.build("Precision")
        assert isinstance(metric, Metric)

    def test_perfect_precision(self) -> None:
        preds, targets = _perfect_preds_and_targets()
        metric = METRICS.build("Precision")
        result = metric(preds, targets)
        assert result == pytest.approx(1.0)

    def test_custom_average(self) -> None:
        metric = METRICS.build("Precision", average="micro")
        assert metric.average == "micro"
        logits, targets = _logits_and_targets()
        result = metric(logits, targets)
        assert 0.0 <= result <= 1.0

    def test_zero_division(self) -> None:
        metric = METRICS.build("Precision", zero_division=1.0)
        assert metric.zero_division == 1.0


# ---------------------------------------------------------------------------
# Recall
# ---------------------------------------------------------------------------


class TestRecall:
    def test_registered(self) -> None:
        assert "Recall" in METRICS

    def test_build(self) -> None:
        metric = METRICS.build("Recall")
        assert isinstance(metric, Metric)

    def test_perfect_recall(self) -> None:
        preds, targets = _perfect_preds_and_targets()
        metric = METRICS.build("Recall")
        result = metric(preds, targets)
        assert result == pytest.approx(1.0)

    def test_custom_average(self) -> None:
        metric = METRICS.build("Recall", average="weighted")
        assert metric.average == "weighted"
        logits, targets = _logits_and_targets()
        result = metric(logits, targets)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# F1Score
# ---------------------------------------------------------------------------


class TestF1Score:
    def test_registered(self) -> None:
        assert "F1Score" in METRICS

    def test_build(self) -> None:
        metric = METRICS.build("F1Score")
        assert isinstance(metric, Metric)

    def test_perfect_f1(self) -> None:
        preds, targets = _perfect_preds_and_targets()
        metric = METRICS.build("F1Score")
        result = metric(preds, targets)
        assert result == pytest.approx(1.0)

    def test_custom_average(self) -> None:
        metric = METRICS.build("F1Score", average="micro")
        assert metric.average == "micro"
        logits, targets = _logits_and_targets()
        result = metric(logits, targets)
        assert 0.0 <= result <= 1.0

    def test_zero_division(self) -> None:
        metric = METRICS.build("F1Score", zero_division=1.0)
        assert metric.zero_division == 1.0


# ---------------------------------------------------------------------------
# AUROC
# ---------------------------------------------------------------------------


class TestAUROC:
    def test_registered(self) -> None:
        assert "AUROC" in METRICS

    def test_build(self) -> None:
        metric = METRICS.build("AUROC")
        assert isinstance(metric, Metric)

    def test_binary_auroc(self) -> None:
        probs, targets = _binary_probs_and_targets()
        metric = METRICS.build("AUROC")
        result = metric(probs, targets)
        assert 0.0 <= result <= 1.0

    def test_binary_perfect(self) -> None:
        targets = torch.tensor([0, 0, 1, 1])
        probs = torch.tensor([0.1, 0.2, 0.8, 0.9])
        metric = METRICS.build("AUROC")
        result = metric(probs, targets)
        assert result == pytest.approx(1.0)

    def test_multiclass_auroc(self) -> None:
        probs, targets = _multiclass_probs_and_targets()
        metric = METRICS.build("AUROC")
        result = metric(probs, targets)
        assert 0.0 <= result <= 1.0

    def test_custom_multi_class(self) -> None:
        metric = METRICS.build("AUROC", multi_class="ovo")
        assert metric.multi_class == "ovo"

    def test_multi_batch_accumulation(self) -> None:
        metric = METRICS.build("AUROC")
        probs1, targets1 = _binary_probs_and_targets()
        probs2, targets2 = _binary_probs_and_targets()
        metric.update(probs1, targets1)
        metric.update(probs2, targets2)
        result = metric.compute()
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# ConfusionMatrix
# ---------------------------------------------------------------------------


class TestConfusionMatrix:
    def test_registered(self) -> None:
        assert "ConfusionMatrix" in METRICS

    def test_build(self) -> None:
        metric = METRICS.build("ConfusionMatrix", num_classes=C)
        assert isinstance(metric, Metric)

    def test_shape(self) -> None:
        logits, targets = _logits_and_targets()
        metric = METRICS.build("ConfusionMatrix", num_classes=C)
        result = metric(logits, targets)
        assert result.shape == (C, C)

    def test_dtype(self) -> None:
        logits, targets = _logits_and_targets()
        metric = METRICS.build("ConfusionMatrix", num_classes=C)
        result = metric(logits, targets)
        assert result.dtype == torch.int64

    def test_perfect_predictions(self) -> None:
        preds, targets = _perfect_preds_and_targets()
        metric = METRICS.build("ConfusionMatrix", num_classes=C)
        result = metric(preds, targets)
        assert result.sum().item() == B
        assert result.diag().sum().item() == B

    def test_row_sums(self) -> None:
        logits, targets = _logits_and_targets()
        metric = METRICS.build("ConfusionMatrix", num_classes=C)
        result = metric(logits, targets)
        assert result.sum().item() == B

    def test_with_1d_preds(self) -> None:
        targets = torch.randint(0, C, (B,))
        preds = targets.clone()
        metric = METRICS.build("ConfusionMatrix", num_classes=C)
        result = metric(preds, targets)
        assert result.diag().sum().item() == B
