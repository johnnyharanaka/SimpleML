"""Trainer class that orchestrates the full training pipeline."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from simpleml.configs.config import Config
from simpleml.metrics.base import Metric

_TRAINING_DEFAULTS: dict[str, Any] = {
    "epochs": 10,
    "batch_size": 32,
    "num_workers": 0,
    "pin_memory": False,
    "device": "auto",
    "grad_clip_norm": None,
    "grad_clip_value": None,
    "mixed_precision": False,
    "checkpoint_dir": "checkpoints",
    "save_every": None,
    "save_best": True,
    "save_last": True,
    "resume_from": None,
    "log_dir": "runs",
    "scheduler_step_on": "epoch",
    "val_every": 1,
}


class Trainer:
    """Orchestrates model training using registry-built components.

    Handles the full training lifecycle: data loading, forward/backward passes,
    gradient clipping, mixed precision, checkpointing, TensorBoard logging,
    and metric evaluation.

    Example::

        trainer = Trainer.from_config("config.yaml")
        results = trainer.fit()
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
        scheduler: Any | None = None,
        metrics: list[Metric] | None = None,
        training_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the trainer with all pipeline components.

        Args:
            model: The neural network to train.
            loss_fn: Loss function module.
            optimizer: Optimizer bound to model parameters.
            train_dataset: Training dataset.
            val_dataset: Optional validation dataset.
            scheduler: Optional learning rate scheduler.
            metrics: Optional list of Metric instances for evaluation.
            training_config: Optional dict overriding training defaults.
        """
        self._cfg = {**_TRAINING_DEFAULTS, **(training_config or {})}

        self.device = self._resolve_device(self._cfg["device"])
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = metrics or []

        self.train_loader = self._create_dataloader(train_dataset, shuffle=True)
        self.val_loader = (
            self._create_dataloader(val_dataset, shuffle=False)
            if val_dataset is not None
            else None
        )

        self.global_step = 0
        self.best_val_loss = math.inf

        self._use_amp = self._cfg["mixed_precision"] and self.device.type == "cuda"
        self._scaler = GradScaler("cuda") if self._use_amp else None

        log_dir = Path(self._cfg["log_dir"])
        self._writer = SummaryWriter(log_dir=str(log_dir))

    @classmethod
    def from_config(cls, config: Config | str | Path) -> Trainer:
        """Build a Trainer from a Config, YAML path, or dict.

        Args:
            config: A Config instance, path to a YAML file, or a dict.

        Returns:
            A fully initialized Trainer.
        """
        if isinstance(config, (str, Path)):
            config = Config.from_yaml(config)
        elif isinstance(config, dict):
            config = Config.from_dict(config)

        model = config.build_model()
        loss_fn = config.build_loss()
        optimizer = config.build_optimizer(model)
        scheduler = config.build_scheduler(optimizer)
        metrics = config.build_metrics()

        train_dataset = config.build_dataset("train")
        val_dataset = (
            config.build_dataset("val")
            if "val" in config.data.get("dataset", {})
            else None
        )

        return cls(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            scheduler=scheduler,
            metrics=metrics,
            training_config=config.training,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> dict[str, Any]:
        """Run the full training loop.

        Returns:
            Dict with keys: ``last_train_loss``, ``last_val_loss``,
            ``best_val_loss``, ``last_metrics``, ``epochs_trained``.
        """
        start_epoch = 0
        if self._cfg["resume_from"] is not None:
            start_epoch = self.load_checkpoint(self._cfg["resume_from"])

        epochs = self._cfg["epochs"]
        last_train_loss = math.nan
        last_val_loss = None
        last_metrics: dict[str, float] = {}

        for epoch in range(start_epoch, epochs):
            last_train_loss = self._train_one_epoch(epoch)
            self._log_metrics({"loss": last_train_loss}, epoch, prefix="train")
            self._log_metrics(
                {"lr": self.optimizer.param_groups[0]["lr"]}, epoch, prefix="train"
            )

            if self.scheduler and self._cfg["scheduler_step_on"] == "epoch":
                self.scheduler.step()

            should_validate = (
                self.val_loader is not None
                and (epoch + 1) % self._cfg["val_every"] == 0
            )
            if should_validate:
                val_result = self._validate_one_epoch(epoch)
                last_val_loss = val_result["loss"]
                last_metrics = val_result.get("metrics", {})
                self._log_metrics({"loss": last_val_loss}, epoch, prefix="val")
                self._log_metrics(last_metrics, epoch, prefix="val")
                self._maybe_save_checkpoint(epoch, last_val_loss)
            else:
                self._maybe_save_checkpoint(epoch, None)

        self._writer.close()

        return {
            "last_train_loss": last_train_loss,
            "last_val_loss": last_val_loss,
            "best_val_loss": (
                self.best_val_loss if self.best_val_loss < math.inf else None
            ),
            "last_metrics": last_metrics,
            "epochs_trained": epochs - start_epoch,
        }

    def validate(self) -> dict[str, Any]:
        """Run standalone validation.

        Returns:
            Dict with ``loss`` and ``metrics`` keys.

        Raises:
            RuntimeError: If no validation DataLoader is configured.
        """
        if self.val_loader is None:
            raise RuntimeError("No validation dataset configured.")
        return self._validate_one_epoch(epoch=0)

    def save_checkpoint(
        self,
        path: str | Path,
        epoch: int,
        val_loss: float | None = None,
    ) -> None:
        """Save a training checkpoint.

        Args:
            path: File path for the checkpoint.
            epoch: Current epoch number.
            val_loss: Optional validation loss at this epoch.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "best_val_loss": self.best_val_loss,
            "val_loss": val_loss,
            "training_config": self._cfg,
        }
        torch.save(state, path)

    def load_checkpoint(self, path: str | Path) -> int:
        """Load a training checkpoint and restore all state.

        Args:
            path: Path to the checkpoint file.

        Returns:
            The epoch to resume from (saved epoch + 1).
        """
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if self.scheduler and state.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.global_step = state.get("global_step", 0)
        self.best_val_loss = state.get("best_val_loss", math.inf)
        return state["epoch"] + 1

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        """Resolve a device string to a torch.device, with MPS priority.

        Args:
            device_str: One of ``"auto"``, ``"mps"``, ``"cuda"``, ``"cpu"``.

        Returns:
            The resolved torch.device.

        Raises:
            ValueError: If the device string is not recognized.
        """
        if device_str == "auto":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        if device_str in ("mps", "cuda", "cpu"):
            return torch.device(device_str)
        raise ValueError(
            f"Unknown device '{device_str}'. Choose from: auto, mps, cuda, cpu."
        )

    def _create_dataloader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        """Create a DataLoader from the training config.

        Args:
            dataset: The dataset to wrap.
            shuffle: Whether to shuffle the data.

        Returns:
            A configured DataLoader.
        """
        return DataLoader(
            dataset,
            batch_size=self._cfg["batch_size"],
            shuffle=shuffle,
            num_workers=self._cfg["num_workers"],
            pin_memory=self._cfg["pin_memory"],
        )

    def _train_one_epoch(self, epoch: int) -> float:
        """Run one training epoch.

        Args:
            epoch: Current epoch index (for logging).

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            inputs, targets = batch[0].to(self.device), batch[1].to(self.device)

            self.optimizer.zero_grad()

            if self._use_amp:
                with autocast("cuda"):
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)
                self._scaler.scale(loss).backward()
                if self._cfg["grad_clip_norm"] is not None:
                    self._scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self._cfg["grad_clip_norm"]
                    )
                if self._cfg["grad_clip_value"] is not None:
                    self._scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_value_(
                        self.model.parameters(), self._cfg["grad_clip_value"]
                    )
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                if self._cfg["grad_clip_norm"] is not None:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self._cfg["grad_clip_norm"]
                    )
                if self._cfg["grad_clip_value"] is not None:
                    nn.utils.clip_grad_value_(
                        self.model.parameters(), self._cfg["grad_clip_value"]
                    )
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            if self.scheduler and self._cfg["scheduler_step_on"] == "step":
                self.scheduler.step()

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate_one_epoch(self, epoch: int) -> dict[str, Any]:
        """Run one validation epoch.

        Args:
            epoch: Current epoch index (for logging).

        Returns:
            Dict with ``loss`` (float) and ``metrics`` (dict of name -> value).
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for m in self.metrics:
            m.reset()

        for batch in self.val_loader:
            inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            total_loss += loss.item()
            num_batches += 1

            for m in self.metrics:
                m.update(outputs, targets)

        avg_loss = total_loss / max(num_batches, 1)
        computed_metrics = {type(m).__name__: m.compute() for m in self.metrics}

        return {"loss": avg_loss, "metrics": computed_metrics}

    def _maybe_save_checkpoint(self, epoch: int, val_loss: float | None) -> None:
        """Conditionally save checkpoints based on config.

        Args:
            epoch: Current epoch index.
            val_loss: Validation loss (None if no validation was run).
        """
        ckpt_dir = Path(self._cfg["checkpoint_dir"])
        epochs = self._cfg["epochs"]

        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            if self._cfg["save_best"]:
                self.save_checkpoint(ckpt_dir / "best.pt", epoch, val_loss)

        if self._cfg["save_every"] and (epoch + 1) % self._cfg["save_every"] == 0:
            self.save_checkpoint(ckpt_dir / f"epoch_{epoch + 1}.pt", epoch, val_loss)

        if self._cfg["save_last"] and (epoch + 1) == epochs:
            self.save_checkpoint(ckpt_dir / "last.pt", epoch, val_loss)

    def _log_metrics(
        self,
        metrics_dict: dict[str, Any],
        step: int,
        prefix: str = "",
    ) -> None:
        """Write scalar metrics to TensorBoard.

        Args:
            metrics_dict: Dict of metric name to value.
            step: Global step or epoch for the x-axis.
            prefix: Tag prefix (e.g., ``"train"`` or ``"val"``).
        """
        for name, value in metrics_dict.items():
            if value is None:
                continue
            tag = f"{prefix}/{name}" if prefix else name
            self._writer.add_scalar(tag, value, step)
