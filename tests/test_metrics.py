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


# ---------------------------------------------------------------------------
# MeanAveragePrecision
# ---------------------------------------------------------------------------


def _det_sample(
    box: list[float],
    score: float,
    label: int,
) -> dict:
    """Build a single-detection prediction dict."""
    return {
        "boxes": torch.tensor([box], dtype=torch.float32),
        "scores": torch.tensor([score]),
        "labels": torch.tensor([label]),
    }


def _gt_sample(box: list[float], label: int) -> dict:
    """Build a single ground-truth dict."""
    return {
        "boxes": torch.tensor([box], dtype=torch.float32),
        "labels": torch.tensor([label]),
    }


class TestMeanAveragePrecision:
    def test_registered(self) -> None:
        assert "MeanAveragePrecision" in METRICS

    def test_build(self) -> None:
        metric = METRICS.build("MeanAveragePrecision")
        assert isinstance(metric, Metric)

    def test_perfect_map(self) -> None:
        pred = _det_sample([0.0, 0.0, 10.0, 10.0], 0.9, 0)
        target = _gt_sample([0.0, 0.0, 10.0, 10.0], 0)
        metric = METRICS.build("MeanAveragePrecision")
        result = metric([pred], [target])
        assert result == pytest.approx(1.0)

    def test_no_overlap_gives_zero(self) -> None:
        pred = _det_sample([100.0, 100.0, 110.0, 110.0], 0.9, 0)
        target = _gt_sample([0.0, 0.0, 10.0, 10.0], 0)
        metric = METRICS.build("MeanAveragePrecision")
        result = metric([pred], [target])
        assert result == pytest.approx(0.0)

    def test_multi_class(self) -> None:
        preds = [
            {
                "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
                "scores": torch.tensor([0.9, 0.8]),
                "labels": torch.tensor([0, 1]),
            }
        ]
        targets = [
            {
                "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
                "labels": torch.tensor([0, 1]),
            }
        ]
        metric = METRICS.build("MeanAveragePrecision")
        result = metric(preds, targets)
        assert result == pytest.approx(1.0)

    def test_multi_batch_accumulation(self) -> None:
        metric = METRICS.build("MeanAveragePrecision")
        pred1 = _det_sample([0.0, 0.0, 10.0, 10.0], 0.9, 0)
        target1 = _gt_sample([0.0, 0.0, 10.0, 10.0], 0)
        pred2 = _det_sample([5.0, 5.0, 15.0, 15.0], 0.8, 0)
        target2 = _gt_sample([5.0, 5.0, 15.0, 15.0], 0)
        metric.update([pred1], [target1])
        metric.update([pred2], [target2])
        result = metric.compute()
        assert result == pytest.approx(1.0)

    def test_custom_iou_threshold(self) -> None:
        # Box with IoU ~0.25 relative to GT — passes 0.2, fails 0.5
        pred = _det_sample([5.0, 0.0, 15.0, 10.0], 0.9, 0)
        target = _gt_sample([0.0, 0.0, 10.0, 10.0], 0)
        metric_low = METRICS.build("MeanAveragePrecision", iou_threshold=0.2)
        metric_high = METRICS.build("MeanAveragePrecision", iou_threshold=0.5)
        assert metric_low([pred], [target]) == pytest.approx(1.0)
        assert metric_high([pred], [target]) == pytest.approx(0.0)

    def test_compute_before_update_raises(self) -> None:
        metric = METRICS.build("MeanAveragePrecision")
        with pytest.raises(RuntimeError, match="called before any update"):
            metric.compute()

    def test_reset_clears_state(self) -> None:
        pred = _det_sample([0.0, 0.0, 10.0, 10.0], 0.9, 0)
        target = _gt_sample([0.0, 0.0, 10.0, 10.0], 0)
        metric = METRICS.build("MeanAveragePrecision")
        metric.update([pred], [target])
        metric.reset()
        with pytest.raises(RuntimeError):
            metric.compute()


# ---------------------------------------------------------------------------
# CorLoc
# ---------------------------------------------------------------------------


class TestCorLoc:
    def test_registered(self) -> None:
        assert "CorLoc" in METRICS

    def test_build(self) -> None:
        metric = METRICS.build("CorLoc")
        assert isinstance(metric, Metric)

    def test_perfect_corloc(self) -> None:
        preds = torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.9]])
        targets = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        metric = METRICS.build("CorLoc")
        assert metric(preds, targets) == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        preds = torch.tensor([[100.0, 100.0, 110.0, 110.0, 0.9]])
        targets = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        metric = METRICS.build("CorLoc")
        assert metric(preds, targets) == pytest.approx(0.0)

    def test_fraction(self) -> None:
        metric = METRICS.build("CorLoc")
        # Image 1: correct
        metric.update(
            torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.9]]),
            torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        )
        # Image 2: miss
        metric.update(
            torch.tensor([[100.0, 100.0, 110.0, 110.0, 0.9]]),
            torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        )
        assert metric.compute() == pytest.approx(0.5)

    def test_empty_preds_is_miss(self) -> None:
        preds = torch.zeros(0, 5)
        targets = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        metric = METRICS.build("CorLoc")
        assert metric(preds, targets) == pytest.approx(0.0)

    def test_top_scoring_box_is_used(self) -> None:
        # Second box has higher score and overlaps; first box does not
        preds = torch.tensor([
            [100.0, 100.0, 110.0, 110.0, 0.3],
            [0.0, 0.0, 10.0, 10.0, 0.9],
        ])
        targets = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        metric = METRICS.build("CorLoc")
        assert metric(preds, targets) == pytest.approx(1.0)

    def test_custom_iou_threshold(self) -> None:
        # Box with IoU ~0.25 relative to GT
        preds = torch.tensor([[5.0, 0.0, 15.0, 10.0, 0.9]])
        targets = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        metric_low = METRICS.build("CorLoc", iou_threshold=0.2)
        metric_high = METRICS.build("CorLoc", iou_threshold=0.5)
        assert metric_low(preds, targets) == pytest.approx(1.0)
        assert metric_high(preds, targets) == pytest.approx(0.0)

    def test_compute_before_update_raises(self) -> None:
        metric = METRICS.build("CorLoc")
        with pytest.raises(RuntimeError, match="called before any update"):
            metric.compute()

    def test_reset_clears_state(self) -> None:
        preds = torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.9]])
        targets = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        metric = METRICS.build("CorLoc")
        metric.update(preds, targets)
        metric.reset()
        with pytest.raises(RuntimeError):
            metric.compute()
