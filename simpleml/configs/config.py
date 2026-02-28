"""Config class for loading YAML/dict configurations and building components."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
import yaml

from simpleml.registries import (
    DATASETS,
    LOSSES,
    METRICS,
    MODELS,
    OPTIMIZERS,
    SCHEDULERS,
)

def _coerce_numbers(obj: Any) -> Any:
    """Recursively convert numeric strings to int or float.

    Allows YAML values like ``1e-3`` or ``"42"`` to be used as numbers even
    when the YAML parser emits them as strings.
    """
    if isinstance(obj, dict):
        return {k: _coerce_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_numbers(v) for v in obj]
    if isinstance(obj, str):
        try:
            return int(obj)
        except ValueError:
            pass
        try:
            return float(obj)
        except ValueError:
            pass
    return obj


_REQUIRED_SECTIONS = {"model", "dataset", "loss", "optimizer"}
_OPTIONAL_SECTIONS = {"scheduler", "metrics", "training", "inference"}
_ALL_SECTIONS = _REQUIRED_SECTIONS | _OPTIONAL_SECTIONS


class Config:
    """Loads a training configuration and builds registered components.

    A Config can be created from a Python dict or a YAML file. It validates
    the structure on creation and provides builder methods that delegate to
    the global registries.

    Example::

        cfg = Config.from_yaml("config.yaml")
        components = cfg.build_all()
    """

    def __init__(self, data: dict[str, Any]) -> None:
        data = _coerce_numbers(data)
        self._validate(data)
        self._data = copy.deepcopy(data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise TypeError(f"YAML root must be a mapping, got {type(data).__name__}")
        return cls(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        """Create a Config from a Python dict."""
        return cls(data)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(data: dict[str, Any]) -> None:
        if not isinstance(data, dict):
            raise TypeError(f"Config data must be a dict, got {type(data).__name__}")

        missing = _REQUIRED_SECTIONS - data.keys()
        if missing:
            raise ValueError(f"Missing required config sections: {sorted(missing)}")

        unknown = data.keys() - _ALL_SECTIONS
        if unknown:
            raise ValueError(f"Unknown config sections: {sorted(unknown)}")

        for section in ("model", "loss", "optimizer"):
            Config._validate_component(data[section], section)

        if "scheduler" in data:
            Config._validate_component(data["scheduler"], "scheduler")

        Config._validate_dataset(data["dataset"])

        if "metrics" in data:
            Config._validate_metrics(data["metrics"])

    @staticmethod
    def _validate_component(section: Any, name: str) -> None:
        if not isinstance(section, dict):
            kind = type(section).__name__
            raise TypeError(f"'{name}' section must be a mapping, got {kind}")
        if "name" not in section:
            raise ValueError(f"'{name}' section must contain a 'name' key")
        if "params" in section and not isinstance(section["params"], dict):
            raise TypeError(f"'{name}.params' must be a mapping")
        if "weights" in section and not isinstance(section["weights"], str):
            raise TypeError(f"'{name}.weights' must be a string path")

    @staticmethod
    def _validate_dataset(section: Any) -> None:
        if not isinstance(section, dict):
            kind = type(section).__name__
            raise TypeError(f"'dataset' section must be a mapping, got {kind}")
        if "train" not in section:
            raise ValueError("'dataset' section must contain a 'train' split")
        for split in ("train", "val", "test"):
            if split in section:
                Config._validate_component(section[split], f"dataset.{split}")

    @staticmethod
    def _validate_metrics(section: Any) -> None:
        if not isinstance(section, list):
            kind = type(section).__name__
            raise TypeError(f"'metrics' section must be a list, got {kind}")
        for i, entry in enumerate(section):
            if not isinstance(entry, dict):
                raise TypeError(f"metrics[{i}] must be a mapping")
            if "name" not in entry:
                raise ValueError(f"metrics[{i}] must contain a 'name' key")
            if "params" in entry and not isinstance(entry["params"], dict):
                raise TypeError(f"metrics[{i}].params must be a mapping")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def data(self) -> dict[str, Any]:
        """Return a deep copy of the raw configuration dict."""
        return copy.deepcopy(self._data)

    @property
    def training(self) -> dict[str, Any]:
        """Return the training section, or an empty dict if absent."""
        return copy.deepcopy(self._data.get("training", {}))

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    def build_model(self) -> Any:
        """Build the model from the registry.

        If the model section contains a ``weights`` key, the corresponding
        file is loaded as a state dict after construction.
        """
        sec = self._data["model"]
        model = MODELS.build(sec["name"], **sec.get("params", {}))
        if "weights" in sec:
            weights_path = Path(sec["weights"])
            if not weights_path.exists():
                raise FileNotFoundError(f"Weights file not found: {weights_path}")
            state_dict = torch.load(weights_path, weights_only=True)
            model.load_state_dict(state_dict)
        return model

    def build_loss(self) -> Any:
        """Build the loss function from the registry."""
        sec = self._data["loss"]
        return LOSSES.build(sec["name"], **sec.get("params", {}))

    def build_dataset(self, split: str = "train") -> Any:
        """Build a dataset for the given split from the registry."""
        ds_section = self._data["dataset"]
        if split not in ds_section:
            raise KeyError(f"Dataset split '{split}' not found in config")
        sec = ds_section[split]
        return DATASETS.build(sec["name"], **sec.get("params", {}))

    def build_optimizer(self, model: Any) -> Any:
        """Build the optimizer, binding to model parameters."""
        sec = self._data["optimizer"]
        params = sec.get("params", {})
        return OPTIMIZERS.build(sec["name"], params=model.parameters(), **params)

    def build_scheduler(self, optimizer: Any) -> Any | None:
        """Build the scheduler, or return None if not configured."""
        if "scheduler" not in self._data:
            return None
        sec = self._data["scheduler"]
        params = sec.get("params", {})
        return SCHEDULERS.build(sec["name"], optimizer=optimizer, **params)

    def build_metrics(self) -> list[Any] | None:
        """Build metrics from the registry, or None if absent."""
        if "metrics" not in self._data:
            return None
        return [
            METRICS.build(entry["name"], **entry.get("params", {}))
            for entry in self._data["metrics"]
        ]

    def build_all(self, model: Any | None = None) -> dict[str, Any]:
        """Build all components and return them as a dict.

        If ``model`` is not provided, it is built from config first.
        """
        if model is None:
            model = self.build_model()
        optimizer = self.build_optimizer(model)
        return {
            "model": model,
            "loss": self.build_loss(),
            "optimizer": optimizer,
            "scheduler": self.build_scheduler(optimizer),
            "metrics": self.build_metrics(),
        }

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        sections = sorted(self._data.keys())
        return f"Config(sections={sections})"
