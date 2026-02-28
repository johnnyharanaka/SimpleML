"""Tests for the simpleml.optimizers module."""

import torch
from torch import nn

from simpleml.registries import OPTIMIZERS, SCHEDULERS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _params() -> list[torch.Tensor]:
    """Return parameters from a small linear layer."""
    return list(nn.Linear(4, 2).parameters())


def _optimizer() -> torch.optim.SGD:
    """Return a basic SGD optimizer for scheduler tests."""
    return torch.optim.SGD(_params(), lr=0.1)


# ---------------------------------------------------------------------------
# SGD
# ---------------------------------------------------------------------------


class TestSGD:
    def test_registered(self) -> None:
        assert "SGD" in OPTIMIZERS

    def test_build(self) -> None:
        opt = OPTIMIZERS.build("SGD", params=_params())
        assert isinstance(opt, torch.optim.SGD)

    def test_step(self) -> None:
        model = nn.Linear(4, 2)
        opt = OPTIMIZERS.build("SGD", params=model.parameters(), lr=0.01)
        loss = model(torch.randn(1, 4)).sum()
        loss.backward()
        opt.step()

    def test_momentum(self) -> None:
        opt = OPTIMIZERS.build("SGD", params=_params(), momentum=0.9)
        assert opt.defaults["momentum"] == 0.9

    def test_nesterov(self) -> None:
        opt = OPTIMIZERS.build("SGD", params=_params(), momentum=0.9, nesterov=True)
        assert opt.defaults["nesterov"] is True


# ---------------------------------------------------------------------------
# Adam
# ---------------------------------------------------------------------------


class TestAdam:
    def test_registered(self) -> None:
        assert "Adam" in OPTIMIZERS

    def test_build(self) -> None:
        opt = OPTIMIZERS.build("Adam", params=_params())
        assert isinstance(opt, torch.optim.Adam)

    def test_step(self) -> None:
        model = nn.Linear(4, 2)
        opt = OPTIMIZERS.build("Adam", params=model.parameters())
        loss = model(torch.randn(1, 4)).sum()
        loss.backward()
        opt.step()

    def test_custom_betas(self) -> None:
        opt = OPTIMIZERS.build("Adam", params=_params(), betas=(0.8, 0.99))
        assert opt.defaults["betas"] == (0.8, 0.99)

    def test_weight_decay(self) -> None:
        opt = OPTIMIZERS.build("Adam", params=_params(), weight_decay=1e-4)
        assert opt.defaults["weight_decay"] == 1e-4


# ---------------------------------------------------------------------------
# AdamW
# ---------------------------------------------------------------------------


class TestAdamW:
    def test_registered(self) -> None:
        assert "AdamW" in OPTIMIZERS

    def test_build(self) -> None:
        opt = OPTIMIZERS.build("AdamW", params=_params())
        assert isinstance(opt, torch.optim.AdamW)

    def test_step(self) -> None:
        model = nn.Linear(4, 2)
        opt = OPTIMIZERS.build("AdamW", params=model.parameters())
        loss = model(torch.randn(1, 4)).sum()
        loss.backward()
        opt.step()

    def test_default_weight_decay(self) -> None:
        opt = OPTIMIZERS.build("AdamW", params=_params())
        assert opt.defaults["weight_decay"] == 1e-2

    def test_amsgrad(self) -> None:
        opt = OPTIMIZERS.build("AdamW", params=_params(), amsgrad=True)
        assert opt.defaults["amsgrad"] is True


# ---------------------------------------------------------------------------
# RMSprop
# ---------------------------------------------------------------------------


class TestRMSprop:
    def test_registered(self) -> None:
        assert "RMSprop" in OPTIMIZERS

    def test_build(self) -> None:
        opt = OPTIMIZERS.build("RMSprop", params=_params())
        assert isinstance(opt, torch.optim.RMSprop)

    def test_step(self) -> None:
        model = nn.Linear(4, 2)
        opt = OPTIMIZERS.build("RMSprop", params=model.parameters())
        loss = model(torch.randn(1, 4)).sum()
        loss.backward()
        opt.step()

    def test_centered(self) -> None:
        opt = OPTIMIZERS.build("RMSprop", params=_params(), centered=True)
        assert opt.defaults["centered"] is True

    def test_alpha(self) -> None:
        opt = OPTIMIZERS.build("RMSprop", params=_params(), alpha=0.9)
        assert opt.defaults["alpha"] == 0.9


# ---------------------------------------------------------------------------
# StepLR
# ---------------------------------------------------------------------------


class TestStepLR:
    def test_registered(self) -> None:
        assert "StepLR" in SCHEDULERS

    def test_build(self) -> None:
        sched = SCHEDULERS.build("StepLR", optimizer=_optimizer(), step_size=5)
        assert isinstance(sched, torch.optim.lr_scheduler.StepLR)

    def test_step(self) -> None:
        opt = _optimizer()
        sched = SCHEDULERS.build("StepLR", optimizer=opt, step_size=1, gamma=0.5)
        lr_before = opt.param_groups[0]["lr"]
        sched.step()
        lr_after = opt.param_groups[0]["lr"]
        assert lr_after < lr_before

    def test_custom_gamma(self) -> None:
        sched = SCHEDULERS.build(
            "StepLR", optimizer=_optimizer(), step_size=10, gamma=0.5
        )
        assert sched.gamma == 0.5


# ---------------------------------------------------------------------------
# CosineAnnealingLR
# ---------------------------------------------------------------------------


class TestCosineAnnealingLR:
    def test_registered(self) -> None:
        assert "CosineAnnealingLR" in SCHEDULERS

    def test_build(self) -> None:
        sched = SCHEDULERS.build("CosineAnnealingLR", optimizer=_optimizer(), T_max=100)
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_step(self) -> None:
        opt = _optimizer()
        sched = SCHEDULERS.build("CosineAnnealingLR", optimizer=opt, T_max=10)
        sched.step()
        assert opt.param_groups[0]["lr"] < 0.1

    def test_eta_min(self) -> None:
        sched = SCHEDULERS.build(
            "CosineAnnealingLR", optimizer=_optimizer(), T_max=10, eta_min=0.01
        )
        assert sched.eta_min == 0.01


# ---------------------------------------------------------------------------
# CosineAnnealingWarmRestarts
# ---------------------------------------------------------------------------


class TestCosineAnnealingWarmRestarts:
    def test_registered(self) -> None:
        assert "CosineAnnealingWarmRestarts" in SCHEDULERS

    def test_build(self) -> None:
        sched = SCHEDULERS.build(
            "CosineAnnealingWarmRestarts", optimizer=_optimizer(), T_0=10
        )
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)

    def test_step(self) -> None:
        opt = _optimizer()
        sched = SCHEDULERS.build("CosineAnnealingWarmRestarts", optimizer=opt, T_0=10)
        sched.step()
        assert opt.param_groups[0]["lr"] <= 0.1

    def test_t_mult(self) -> None:
        sched = SCHEDULERS.build(
            "CosineAnnealingWarmRestarts",
            optimizer=_optimizer(),
            T_0=10,
            T_mult=2,
        )
        assert sched.T_mult == 2


# ---------------------------------------------------------------------------
# OneCycleLR
# ---------------------------------------------------------------------------


class TestOneCycleLR:
    def test_registered(self) -> None:
        assert "OneCycleLR" in SCHEDULERS

    def test_build(self) -> None:
        sched = SCHEDULERS.build(
            "OneCycleLR", optimizer=_optimizer(), max_lr=0.1, total_steps=100
        )
        assert isinstance(sched, torch.optim.lr_scheduler.OneCycleLR)

    def test_step(self) -> None:
        opt = _optimizer()
        sched = SCHEDULERS.build(
            "OneCycleLR", optimizer=opt, max_lr=0.1, total_steps=100
        )
        sched.step()

    def test_epochs_and_steps_per_epoch(self) -> None:
        sched = SCHEDULERS.build(
            "OneCycleLR",
            optimizer=_optimizer(),
            max_lr=0.1,
            epochs=10,
            steps_per_epoch=50,
        )
        assert isinstance(sched, torch.optim.lr_scheduler.OneCycleLR)

    def test_anneal_strategy(self) -> None:
        sched = SCHEDULERS.build(
            "OneCycleLR",
            optimizer=_optimizer(),
            max_lr=0.1,
            total_steps=100,
            anneal_strategy="linear",
        )
        assert sched._anneal_func is not None
