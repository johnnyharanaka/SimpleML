"""COCO-style and per-image (Labelme) classification dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

from simpleml.registries import DATASETS

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@DATASETS.register
class COCOClassificationDataset(Dataset):
    """Classification dataset built from a directory with ``images/`` and ``annotations/``.

    Supports two annotation formats automatically detected at load time:

    - **Per-image JSON** (Labelme): one JSON per image inside ``annotations/``,
      each with a ``"shapes"`` list. The first shape's ``label`` becomes the
      image class. Images with no matching annotation file use ``default_class``.
    - **COCO JSON**: a single JSON file inside ``annotations/`` with ``"images"``,
      ``"annotations"``, and ``"categories"`` keys. The first annotation's
      category name becomes the image class.

    The raw annotation data for each sample is stored in :attr:`annotations` as
    a list of dicts, making bounding box information accessible for future
    weakly-supervised detection (WSOD) use.

    Expected layout::

        root/
            images/
                img001.jpg
            annotations/
                img001.json   # per-image (Labelme), OR
                _annotations.coco.json  # single COCO file

    Example config::

        dataset:
          val:
            name: COCOClassificationDataset
            params:
              root: data/data_fire/Val
              classes: ["Fire", "Not"]
              default_class: Not
    """

    def __init__(
        self,
        root: str,
        transform: A.Compose | list[dict] | None = None,
        classes: list[str] | None = None,
        default_class: str | None = None,
        images_dir: str = "images",
        annotations_dir: str = "annotations",
    ) -> None:
        """Initialize the dataset.

        Args:
            root: Root directory containing ``images_dir`` and ``annotations_dir``.
            transform: Albumentations ``Compose`` pipeline or list of transform
                dicts (same format as :class:`ImageFolderDataset`).
            classes: Explicit ordered list of class names. When provided the
                class-to-index mapping is fixed to this order, which keeps
                labels consistent with a paired :class:`ImageFolderDataset`
                training split.
            default_class: Class name assigned to images with no annotation
                file (per-image format only). Images without an annotation are
                skipped when ``None``.
            images_dir: Subdirectory name for images. Defaults to ``"images"``.
            annotations_dir: Subdirectory name for annotations. Defaults to
                ``"annotations"``.
        """
        self.root = Path(root)
        images_path = self.root / images_dir
        annotations_path = self.root / annotations_dir

        self.transform = (
            self._build_transform(transform)
            if isinstance(transform, list)
            else transform
        )
        self._to_tensor = A.Compose([ToTensorV2()])

        json_files = sorted(annotations_path.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(
                f"No JSON annotation files found in {annotations_path}"
            )

        if len(json_files) == 1:
            raw_samples, derived_classes = self._load_coco(images_path, json_files[0])
        else:
            raw_samples, derived_classes = self._load_per_image(
                images_path, annotations_path, default_class
            )

        self._classes = list(classes) if classes is not None else sorted(derived_classes)
        self._class_to_idx = {cls: idx for idx, cls in enumerate(self._classes)}

        valid = [
            (path, self._class_to_idx[label], ann)
            for path, label, ann in raw_samples
            if label in self._class_to_idx
        ]
        if not valid:
            raise FileNotFoundError(f"No valid labeled samples found in {self.root}")

        self._samples: list[tuple[Path, int]] = [(p, lbl) for p, lbl, _ in valid]
        self.annotations: list[list[dict[str, Any]]] = [ann for _, _, ann in valid]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def classes(self) -> list[str]:
        """Ordered list of class names."""
        return list(self._classes)

    @property
    def num_classes(self) -> int:
        """Number of classes."""
        return len(self._classes)

    @property
    def class_to_idx(self) -> dict[str, int]:
        """Mapping from class name to integer label."""
        return dict(self._class_to_idx)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Return ``(image_tensor, label)`` for the given index."""
        path, label = self._samples[index]

        image = Image.open(path).convert("RGB")
        image = np.array(image)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        if not isinstance(image, torch.Tensor):
            image = self._to_tensor(image=image)["image"]

        return image, label

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_coco(
        images_path: Path,
        annotation_file: Path,
    ) -> tuple[list[tuple[Path, str, list[dict]]], set[str]]:
        """Load samples from a single COCO-format JSON file."""
        data = json.loads(annotation_file.read_text())
        categories = {cat["id"]: cat["name"] for cat in data.get("categories", [])}

        img_to_anns: dict[int, list[dict]] = {}
        for ann in data.get("annotations", []):
            img_to_anns.setdefault(ann["image_id"], []).append(ann)

        samples: list[tuple[Path, str, list[dict]]] = []
        all_classes: set[str] = set(categories.values())

        for img_info in data.get("images", []):
            img_path = images_path / img_info["file_name"]
            if not img_path.exists():
                continue
            anns = img_to_anns.get(img_info["id"], [])
            if not anns:
                continue
            label = categories[anns[0]["category_id"]]
            samples.append((img_path, label, anns))

        return samples, all_classes

    @staticmethod
    def _load_per_image(
        images_path: Path,
        annotations_path: Path,
        default_class: str | None,
    ) -> tuple[list[tuple[Path, str, list[dict]]], set[str]]:
        """Load samples from per-image Labelme-format JSON files."""
        ann_by_stem = {p.stem: p for p in annotations_path.glob("*.json")}

        samples: list[tuple[Path, str, list[dict]]] = []
        all_classes: set[str] = set()

        for img_path in sorted(images_path.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in _IMAGE_EXTENSIONS:
                continue

            ann_file = ann_by_stem.get(img_path.stem)
            if ann_file is None:
                if default_class is not None:
                    all_classes.add(default_class)
                    samples.append((img_path, default_class, []))
                continue

            data = json.loads(ann_file.read_text())
            shapes = data.get("shapes", [])
            label = shapes[0]["label"] if shapes else default_class
            if label is None:
                continue
            all_classes.add(label)
            samples.append((img_path, label, shapes))

        return samples, all_classes

    @staticmethod
    def _build_transform(transform_list: list[dict]) -> A.Compose:
        """Build an ``A.Compose`` pipeline from a list of transform dicts."""
        transforms = []
        for entry in transform_list:
            name = entry["name"]
            params = entry.get("params", {})
            if name == "ToTensorV2":
                transforms.append(ToTensorV2(**params))
            else:
                cls = getattr(A, name)
                transforms.append(cls(**params))
        return A.Compose(transforms)