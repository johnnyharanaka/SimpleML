"""Tests for the simpleml.losses module."""

import torch
from torch import nn

from simpleml.registries import LOSSES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

B, C, D = 8, 4, 16


def _logits_and_targets() -> tuple[torch.Tensor, torch.Tensor]:
    """Return classification logits (B, C) and integer targets (B,)."""
    return torch.randn(B, C), torch.randint(0, C, (B,))


def _binary_logits_and_targets() -> tuple[torch.Tensor, torch.Tensor]:
    """Return binary logits (B, C) and float targets (B, C)."""
    return torch.randn(B, C), torch.rand(B, C).round()


def _embeddings(n: int = B) -> torch.Tensor:
    """Return random embeddings (n, D)."""
    return torch.randn(n, D)


# ---------------------------------------------------------------------------
# CrossEntropyLoss
# ---------------------------------------------------------------------------


class TestCrossEntropyLoss:
    def test_registered(self) -> None:
        assert "CrossEntropyLoss" in LOSSES

    def test_build(self) -> None:
        loss_fn = LOSSES.build("CrossEntropyLoss")
        assert isinstance(loss_fn, nn.Module)

    def test_forward_scalar(self) -> None:
        loss_fn = LOSSES.build("CrossEntropyLoss")
        logits, targets = _logits_and_targets()
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0

    def test_label_smoothing(self) -> None:
        loss_fn = LOSSES.build("CrossEntropyLoss", label_smoothing=0.1)
        logits, targets = _logits_and_targets()
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0


# ---------------------------------------------------------------------------
# BCEWithLogitsLoss
# ---------------------------------------------------------------------------


class TestBCEWithLogitsLoss:
    def test_registered(self) -> None:
        assert "BCEWithLogitsLoss" in LOSSES

    def test_build(self) -> None:
        loss_fn = LOSSES.build("BCEWithLogitsLoss")
        assert isinstance(loss_fn, nn.Module)

    def test_forward_scalar(self) -> None:
        loss_fn = LOSSES.build("BCEWithLogitsLoss")
        logits, targets = _binary_logits_and_targets()
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0

    def test_pos_weight(self) -> None:
        pw = torch.ones(C) * 2.0
        loss_fn = LOSSES.build("BCEWithLogitsLoss", pos_weight=pw)
        logits, targets = _binary_logits_and_targets()
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0


# ---------------------------------------------------------------------------
# FocalLoss
# ---------------------------------------------------------------------------


class TestFocalLoss:
    def test_registered(self) -> None:
        assert "FocalLoss" in LOSSES

    def test_build(self) -> None:
        loss_fn = LOSSES.build("FocalLoss")
        assert isinstance(loss_fn, nn.Module)

    def test_forward_scalar(self) -> None:
        loss_fn = LOSSES.build("FocalLoss")
        logits, targets = _logits_and_targets()
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0

    def test_custom_gamma(self) -> None:
        loss_fn = LOSSES.build("FocalLoss", gamma=5.0)
        assert loss_fn.gamma == 5.0
        logits, targets = _logits_and_targets()
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0

    def test_reduction_none(self) -> None:
        loss_fn = LOSSES.build("FocalLoss", reduction="none")
        logits, targets = _logits_and_targets()
        loss = loss_fn(logits, targets)
        assert loss.shape == (B,)

    def test_reduction_sum(self) -> None:
        loss_fn = LOSSES.build("FocalLoss", reduction="sum")
        logits, targets = _logits_and_targets()
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0


# ---------------------------------------------------------------------------
# SupConLoss
# ---------------------------------------------------------------------------


class TestSupConLoss:
    def test_registered(self) -> None:
        assert "SupConLoss" in LOSSES

    def test_build(self) -> None:
        loss_fn = LOSSES.build("SupConLoss")
        assert isinstance(loss_fn, nn.Module)

    def test_forward_two_views(self) -> None:
        loss_fn = LOSSES.build("SupConLoss")
        features = torch.randn(B, 2, D)
        labels = torch.randint(0, 3, (B,))
        loss = loss_fn(features, labels)
        assert loss.dim() == 0

    def test_forward_single_view(self) -> None:
        loss_fn = LOSSES.build("SupConLoss")
        features = torch.randn(B, D)
        labels = torch.randint(0, 3, (B,))
        loss = loss_fn(features, labels)
        assert loss.dim() == 0

    def test_custom_temperature(self) -> None:
        loss_fn = LOSSES.build("SupConLoss", temperature=0.1)
        assert loss_fn.temperature == 0.1


# ---------------------------------------------------------------------------
# NTXentLoss
# ---------------------------------------------------------------------------


class TestNTXentLoss:
    def test_registered(self) -> None:
        assert "NTXentLoss" in LOSSES

    def test_build(self) -> None:
        loss_fn = LOSSES.build("NTXentLoss")
        assert isinstance(loss_fn, nn.Module)

    def test_forward_scalar(self) -> None:
        loss_fn = LOSSES.build("NTXentLoss")
        z_i, z_j = _embeddings(), _embeddings()
        loss = loss_fn(z_i, z_j)
        assert loss.dim() == 0

    def test_custom_temperature(self) -> None:
        loss_fn = LOSSES.build("NTXentLoss", temperature=0.1)
        assert loss_fn.temperature == 0.1

    def test_positive_loss(self) -> None:
        loss_fn = LOSSES.build("NTXentLoss")
        z_i, z_j = _embeddings(), _embeddings()
        loss = loss_fn(z_i, z_j)
        assert loss.item() > 0


# ---------------------------------------------------------------------------
# TripletMarginLoss
# ---------------------------------------------------------------------------


class TestTripletMarginLoss:
    def test_registered(self) -> None:
        assert "TripletMarginLoss" in LOSSES

    def test_build(self) -> None:
        loss_fn = LOSSES.build("TripletMarginLoss")
        assert isinstance(loss_fn, nn.Module)

    def test_forward_scalar(self) -> None:
        loss_fn = LOSSES.build("TripletMarginLoss")
        anchor, positive, negative = _embeddings(), _embeddings(), _embeddings()
        loss = loss_fn(anchor, positive, negative)
        assert loss.dim() == 0

    def test_custom_margin(self) -> None:
        loss_fn = LOSSES.build("TripletMarginLoss", margin=2.0)
        assert loss_fn.margin == 2.0

    def test_custom_p(self) -> None:
        loss_fn = LOSSES.build("TripletMarginLoss", p=1.0)
        assert loss_fn.p == 1.0
