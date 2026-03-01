"""Predictor class for running inference with trained models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from simpleml.metrics.base import Metric
from simpleml.registries import DATASETS, METRICS, MODELS

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

_INFERENCE_DEFAULTS: dict[str, Any] = {
    "batch_size": 32,
    "num_workers": 0,
    "device": "auto",
}


@dataclass
class PredictionResult:
    """Container for model prediction outputs.

    Attributes:
        logits: Raw model output of shape ``(N, C)``.
        probabilities: Softmax probabilities of shape ``(N, C)``.
        predicted_classes: Argmax class indices of shape ``(N,)``.
    """

    logits: Tensor
    probabilities: Tensor
    predicted_classes: Tensor


class Predictor:
    """Run inference with a trained model.

    Supports single-image, batch, and dataset-level prediction, as well as
    evaluation with registered metrics.

    Example::

        predictor = Predictor.from_config("config.yaml", checkpoint="best.pt")
        result = predictor.predict_image("photo.jpg")
        print(result.predicted_classes)
    """

    def __init__(
        self,
        model: nn.Module,
        transform: Any | None = None,
        metrics: list[Metric] | None = None,
        inference_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the Predictor.

        Args:
            model: A trained ``nn.Module``.
            transform: Optional Albumentations transform pipeline.
            metrics: Optional list of Metric instances for evaluation.
            inference_config: Optional dict overriding inference defaults
                (``batch_size``, ``num_workers``, ``device``).
        """
        self._cfg = {**_INFERENCE_DEFAULTS, **(inference_config or {})}
        self.device = self._resolve_device(self._cfg["device"])
        self.model = model.to(self.device)
        self.model.eval()
        self.transform = transform
        self.metrics = metrics or []
        self._to_tensor = ToTensorV2()
        self._raw_config: dict[str, Any] | None = None

    @classmethod
    def from_config(
        cls,
        config: str | Path | dict[str, Any],
        checkpoint: str | Path | None = None,
    ) -> Predictor:
        """Build a Predictor from a YAML path or dict.

        Only the ``model`` section is required. Sections like ``loss`` and
        ``optimizer`` are ignored. An optional ``inference`` section can
        override batch_size, device, and num_workers. An optional ``metrics``
        section enables :meth:`evaluate`.

        Args:
            config: Path to a YAML file or a raw config dict.
            checkpoint: Optional path to a checkpoint file. Accepts both
                Trainer checkpoints (with ``model_state_dict`` key) and
                plain state dicts.

        Returns:
            A fully initialized Predictor.

        Raises:
            ValueError: If the config has no ``model`` section.
        """
        import yaml

        if isinstance(config, (str, Path)):
            path = Path(config)
            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {path}")
            with open(path) as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                raise TypeError(
                    f"YAML root must be a mapping, got {type(data).__name__}"
                )
        else:
            data = config

        if "model" not in data:
            raise ValueError("Config must contain a 'model' section")

        model_sec = data["model"]
        model = MODELS.build(model_sec["name"], **model_sec.get("params", {}))

        if checkpoint is not None:
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            else:
                model.load_state_dict(ckpt)
        elif "weights" in model_sec:
            weights_path = Path(model_sec["weights"])
            if not weights_path.exists():
                raise FileNotFoundError(f"Weights file not found: {weights_path}")
            state_dict = torch.load(weights_path, weights_only=True)
            model.load_state_dict(state_dict)

        inference_cfg = {**_INFERENCE_DEFAULTS, **data.get("inference", {})}

        metrics: list[Metric] | None = None
        if "metrics" in data:
            metrics = [
                METRICS.build(entry["name"], **entry.get("params", {}))
                for entry in data["metrics"]
            ]

        predictor = cls(
            model=model,
            metrics=metrics,
            inference_config=inference_cfg,
        )
        predictor._raw_config = data
        return predictor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_image(self, image: Image.Image | str | Path) -> PredictionResult:
        """Run prediction on a single image.

        Args:
            image: A PIL Image, or a file path (str / Path) to an image.

        Returns:
            A PredictionResult with batch dimension of 1.
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        arr = np.array(image.convert("RGB"))

        if self.transform is not None:
            tensor = self.transform(image=arr)["image"]
        else:
            tensor = self._to_tensor(image=arr)["image"]

        tensor = tensor.float().unsqueeze(0).to(self.device)
        return self._predict_tensor(tensor)

    def predict_batch(self, images: list[str | Path] | str | Path) -> PredictionResult:
        """Run prediction on a batch of images.

        Args:
            images: A list of image file paths, or a directory path. When a
                directory is given, it is globbed for common image extensions.

        Returns:
            A PredictionResult covering all images.
        """
        if isinstance(images, (str, Path)):
            dir_path = Path(images)
            paths = sorted(
                p
                for p in dir_path.iterdir()
                if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
            )
        else:
            paths = [Path(p) for p in images]

        tensors: list[Tensor] = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            arr = np.array(img)
            if self.transform is not None:
                t = self.transform(image=arr)["image"]
            else:
                t = self._to_tensor(image=arr)["image"]
            tensors.append(t)

        stacked = torch.stack(tensors).float()
        dataset = TensorDataset(stacked)
        loader = DataLoader(
            dataset,
            batch_size=self._cfg["batch_size"],
            shuffle=False,
            num_workers=self._cfg["num_workers"],
        )

        all_logits: list[Tensor] = []
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                logits = self.model(batch)
                all_logits.append(logits.cpu())

        logits = torch.cat(all_logits, dim=0)
        probs = torch.softmax(logits, dim=1)
        classes = torch.argmax(logits, dim=1)
        return PredictionResult(
            logits=logits, probabilities=probs, predicted_classes=classes
        )

    def predict_dataset(self, dataset: Dataset) -> PredictionResult:
        """Run prediction over an entire dataset.

        Args:
            dataset: A PyTorch Dataset returning ``(input, ...)`` tuples.

        Returns:
            A PredictionResult covering all samples.
        """
        loader = DataLoader(
            dataset,
            batch_size=self._cfg["batch_size"],
            shuffle=False,
            num_workers=self._cfg["num_workers"],
        )

        all_logits: list[Tensor] = []
        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].float().to(self.device)
                logits = self.model(inputs)
                all_logits.append(logits.cpu())

        logits = torch.cat(all_logits, dim=0)
        probs = torch.softmax(logits, dim=1)
        classes = torch.argmax(logits, dim=1)
        return PredictionResult(
            logits=logits, probabilities=probs, predicted_classes=classes
        )

    def evaluate(self, dataset: Dataset) -> dict[str, Any]:
        """Run prediction and compute metrics on a labeled dataset.

        Dispatches to :meth:`_evaluate_detection` when the model exposes a
        ``detect`` method, otherwise to :meth:`_evaluate_classification`.

        Args:
            dataset: A PyTorch Dataset returning ``(input, target)`` tuples.

        Returns:
            Dict with ``predictions`` (PredictionResult) and ``metrics``
            (dict mapping metric name to computed value).
        """
        collate_fn = getattr(dataset, 'collate_fn', None)
        loader = DataLoader(
            dataset,
            batch_size=self._cfg["batch_size"],
            shuffle=False,
            num_workers=self._cfg["num_workers"],
            collate_fn=collate_fn,
        )

        for m in self.metrics:
            m.reset()

        if hasattr(self.model, 'detect'):
            all_logits = self._evaluate_detection(loader)
        else:
            all_logits = self._evaluate_classification(loader)

        logits = torch.cat(all_logits, dim=0)
        probs = torch.softmax(logits, dim=1)
        classes = torch.argmax(logits, dim=1)
        computed_metrics = {type(m).__name__: m.compute() for m in self.metrics}

        return {
            "predictions": PredictionResult(
                logits=logits, probabilities=probs, predicted_classes=classes
            ),
            "metrics": computed_metrics,
        }

    def _evaluate_detection(self, loader: DataLoader) -> list[Tensor]:
        """Evaluate loop for object-detection models.

        Expects the model to expose a ``detect(inputs)`` method that returns
        bounding boxes per image alongside the standard forward logits.

        Args:
            loader: DataLoader whose batches are ``(inputs, targets)`` where
                targets are dicts with ``boxes`` and ``labels`` keys.

        Returns:
            List of logit tensors (one per batch) on CPU.
        """
        from simpleml.metrics.corloc import CorLoc
        from simpleml.metrics.mean_average_precision import MeanAveragePrecision

        all_logits: list[Tensor] = []
        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].float().to(self.device)
                raw_targets = batch[1]

                logits = self.model(inputs)
                all_logits.append(logits.cpu())
                boxes_per_image = self.model.detect(inputs)
                pred_classes = logits.argmax(dim=1)

                preds_map, targets_map = [], []
                for i, (bboxes, target) in enumerate(zip(boxes_per_image, raw_targets)):
                    cls = int(pred_classes[i].item())
                    if bboxes:
                        b_t = torch.tensor([[b[0], b[1], b[2], b[3]] for b in bboxes], dtype=torch.float32)
                        s_t = torch.tensor([b[4] for b in bboxes], dtype=torch.float32)
                        l_t = torch.full((len(bboxes),), cls, dtype=torch.long)
                    else:
                        b_t = torch.zeros(0, 4)
                        s_t = torch.zeros(0)
                        l_t = torch.zeros(0, dtype=torch.long)
                    corloc_t = torch.cat([b_t, s_t.unsqueeze(1)], dim=1)
                    gt_boxes = target['boxes']
                    preds_map.append({'boxes': b_t, 'scores': s_t, 'labels': l_t})
                    targets_map.append({'boxes': gt_boxes, 'labels': target['labels']})
                    for m in self.metrics:
                        if isinstance(m, CorLoc):
                            m.update(corloc_t, gt_boxes)
                for m in self.metrics:
                    if isinstance(m, MeanAveragePrecision):
                        m.update(preds_map, targets_map)

        return all_logits

    def _evaluate_classification(self, loader: DataLoader) -> list[Tensor]:
        """Evaluate loop for classification models.

        Args:
            loader: DataLoader whose batches are ``(inputs, targets)`` where
                targets are integer class indices.

        Returns:
            List of logit tensors (one per batch) on CPU.
        """
        all_logits: list[Tensor] = []
        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].float().to(self.device)
                targets = batch[1].to(self.device)
                logits = self.model(inputs)
                all_logits.append(logits.cpu())
                for m in self.metrics:
                    m.update(logits, targets)
        return all_logits

    def evaluate_from_config(self, split: str = "test") -> dict[str, Any]:
        """Build a dataset from the stored config and evaluate on it.

        Args:
            split: Dataset split to use (``"test"``, ``"val"``, or ``"train"``).

        Returns:
            Same as :meth:`evaluate`.

        Raises:
            RuntimeError: If no config was stored (Predictor not built via
                :meth:`from_config`).
            KeyError: If the requested split is not in the config.
        """
        if self._raw_config is None:
            raise RuntimeError(
                "No config stored. Use Predictor.from_config() to enable "
                "evaluate_from_config()."
            )
        ds_section = self._raw_config.get("dataset", {})
        if split not in ds_section:
            raise KeyError(f"Dataset split '{split}' not found in config")
        sec = ds_section[split]
        dataset = DATASETS.build(sec["name"], **sec.get("params", {}))
        return self.evaluate(dataset)

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

    @torch.no_grad()
    def _predict_tensor(self, tensor: Tensor) -> PredictionResult:
        """Run the model on a prepared tensor batch.

        Args:
            tensor: Input tensor of shape ``(N, C, H, W)`` already on device.

        Returns:
            A PredictionResult with logits, probabilities, and classes.
        """
        logits = self.model(tensor).cpu()
        probs = torch.softmax(logits, dim=1)
        classes = torch.argmax(logits, dim=1)
        return PredictionResult(
            logits=logits, probabilities=probs, predicted_classes=classes
        )
