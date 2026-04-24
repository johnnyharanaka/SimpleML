"""Fluent API for building and running ML experiments without touching raw dicts."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

_RESERVED = frozenset(
    {
        "data",
        "metrics",
        "train_config",
        "infer_config",
        "fit",
        "evaluate",
        "predict_image",
        "predict_batch",
        "from_yaml",
        "to_config",
    }
)


class API:
    """Fluent builder that accumulates a config dict and delegates to Trainer/Predictor.

    Any attribute access that is not a reserved method returns a setter that
    writes a ``{"name": ..., "params": {...}}`` entry into the config under that
    section name. This means adding new registry modules never requires changes
    here.

    Example::

        from simpleml import API

        results = (
            API()
            .model("ResNet", variant="resnet18", pretrained=True, num_classes=10)
            .loss("CrossEntropyLoss")
            .optimizer("AdamW", lr=1e-3)
            .data(train="data/train", val="data/val")
            .train_config(epochs=10, device="mps")
            .fit()
        )
    """

    def __init__(self) -> None:
        self._config: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Generic section setter via __getattr__
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or name in _RESERVED:
            raise AttributeError(name)

        def _set_section(component_name: str, **params: Any) -> API:
            self._config[name] = (
                {"name": component_name, "params": params}
                if params
                else {"name": component_name}
            )
            return self

        return _set_section

    # ------------------------------------------------------------------
    # Special-case configuration methods
    # ------------------------------------------------------------------

    def data(
        self,
        train: str | dict | None = None,
        val: str | dict | None = None,
        test: str | dict | None = None,
        dataset: str = "ImageFolderDataset",
    ) -> API:
        """Set the dataset section.

        Each split can be a string (shorthand for ``ImageFolderDataset`` with
        ``root=<string>``) or a full component dict (``{"name": ..., "params": {...}}``).

        Args:
            train: Training split path or component dict.
            val: Validation split path or component dict.
            test: Test split path or component dict.
            dataset: Default dataset class name used when a split is given as a
                string. Defaults to ``"ImageFolderDataset"``.

        Returns:
            self
        """
        ds: dict[str, Any] = {}
        for split_name, split_value in (("train", train), ("val", val), ("test", test)):
            if split_value is None:
                continue
            if isinstance(split_value, str):
                ds[split_name] = {"name": dataset, "params": {"root": split_value}}
            else:
                ds[split_name] = split_value
        self._config["dataset"] = ds
        return self

    def metrics(self, *names_or_configs: str | dict) -> API:
        """Set the metrics list.

        Each entry can be a string (shorthand for ``{"name": <string>}``) or a
        full component dict (``{"name": ..., "params": {...}}``).

        Args:
            *names_or_configs: Metric names and/or dicts.

        Returns:
            self
        """
        self._config["metrics"] = [
            {"name": item} if isinstance(item, str) else item
            for item in names_or_configs
        ]
        return self

    def train_config(self, **kwargs: Any) -> API:
        """Merge keyword arguments into the ``training`` config section.

        Args:
            **kwargs: Training settings such as ``epochs``, ``batch_size``,
                ``device``, ``save_best``, etc.

        Returns:
            self
        """
        training = self._config.get("training", {})
        training.update(kwargs)
        self._config["training"] = training
        return self

    def infer_config(self, **kwargs: Any) -> API:
        """Merge keyword arguments into the ``inference`` config section.

        Args:
            **kwargs: Inference settings such as ``batch_size``, ``device``,
                ``num_workers``.

        Returns:
            self
        """
        inference = self._config.get("inference", {})
        inference.update(kwargs)
        self._config["inference"] = inference
        return self

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> API:
        """Load a YAML config file and populate a new API instance.

        Args:
            path: Path to the YAML file.

        Returns:
            A new API instance with ``_config`` populated from the file.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise TypeError(f"YAML root must be a mapping, got {type(data).__name__}")
        instance = cls()
        instance._config = data
        return instance

    def to_config(self) -> dict[str, Any]:
        """Return a deep copy of the accumulated config dict.

        Returns:
            A copy of the internal config, suitable for inspection or export.
        """
        return copy.deepcopy(self._config)

    # ------------------------------------------------------------------
    # Execution methods
    # ------------------------------------------------------------------

    def fit(
        self,
        epochs: int | None = None,
        best: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build a Trainer from the accumulated config and run training.

        Args:
            epochs: If given, overrides ``training.epochs`` for this run only
                (does not mutate the stored config).
            best: If given, overrides the filename used when saving the best
                checkpoint (e.g. ``"resnet.pth"``).
            **kwargs: Additional training settings merged on top of
                ``training`` for this run only.

        Returns:
            Dict returned by :meth:`Trainer.fit`.
        """
        from simpleml.trainers import Trainer

        cfg = copy.deepcopy(self._config)
        if epochs is not None or best is not None or kwargs:
            training = cfg.get("training", {})
            if epochs is not None:
                training["epochs"] = epochs
            if best is not None:
                training["best_filename"] = best
            training.update(kwargs)
            cfg["training"] = training

        return Trainer.from_config(cfg).fit()

    def evaluate(
        self,
        split: str = "test",
        checkpoint: str | Path | None = None,
    ) -> dict[str, Any]:
        """Build a Predictor and evaluate on the given dataset split.

        Args:
            split: Dataset split to evaluate on (``"test"``, ``"val"``, or
                ``"train"``).
            checkpoint: Optional path to a checkpoint file. If provided,
                model weights are loaded from it.

        Returns:
            Dict returned by :meth:`Predictor.evaluate_from_config`.
        """
        from simpleml.inference import Predictor

        cfg = copy.deepcopy(self._config)
        return Predictor.from_config(cfg, checkpoint=checkpoint).evaluate_from_config(
            split
        )

    def predict_image(
        self,
        image: Any,
        checkpoint: str | Path | None = None,
    ) -> Any:
        """Run inference on a single image.

        Args:
            image: A PIL Image or a file path (str / Path).
            checkpoint: Optional path to a checkpoint file.

        Returns:
            A :class:`PredictionResult` with batch dimension of 1.
        """
        from simpleml.inference import Predictor

        cfg = copy.deepcopy(self._config)
        return Predictor.from_config(cfg, checkpoint=checkpoint).predict_image(image)

    def predict_batch(
        self,
        images: Any,
        checkpoint: str | Path | None = None,
    ) -> Any:
        """Run inference on a batch of images.

        Args:
            images: A list of image file paths, or a directory path.
            checkpoint: Optional path to a checkpoint file.

        Returns:
            A :class:`PredictionResult` covering all images.
        """
        from simpleml.inference import Predictor

        cfg = copy.deepcopy(self._config)
        return Predictor.from_config(cfg, checkpoint=checkpoint).predict_batch(images)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        sections = sorted(self._config.keys())
        return f"API(sections={sections})"
