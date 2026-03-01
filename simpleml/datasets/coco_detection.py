"""COCO detection dataset returning images alongside bounding box annotations."""

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
class COCODetectionDataset(Dataset):
    """Detection dataset loaded from a single COCO-format JSON file.

    Returns ``(image, target)`` per sample, where ``target`` is a dict with:

    - ``boxes``       — ``FloatTensor[N, 4]`` in ``[x1, y1, x2, y2]`` format.
    - ``labels``      — ``LongTensor[N]`` per-box category index.
    - ``image_label`` — ``int`` image-level label for WSOD training supervision.
    - ``image_id``    — ``int`` original COCO image id.

    The image-level label is the category of the first annotation. Images with
    no annotations are skipped unless ``default_class`` is set, in which case
    they are included with an empty box list and that class as ``image_label``.

    Expected layout::

        root/
            images/
                img001.jpg
            annotations/
                _annotations.coco.json

    Bounding box transforms are applied via Albumentations ``BboxParams`` so
    boxes are consistently updated alongside image augmentations. Boxes that
    become fully invisible after cropping/flipping are dropped automatically.

    Example config::

        dataset:
          val:
            name: COCODetectionDataset
            params:
              root: data/data_fire/Val
              classes: ["Fire", "Not"]

    For batched loading, pass ``collate_fn=COCODetectionDataset.collate_fn``
    to ``DataLoader``.
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
            transform: Albumentations ``Compose`` (must include ``bbox_params``)
                or a list of transform dicts. When a list is passed, bbox-aware
                params are added automatically.
            classes: Explicit ordered list of class names. Fixes the
                class-to-index mapping across train/val/test splits.
            default_class: Class assigned to images with no annotations.
                Images without annotations are skipped when ``None``.
            images_dir: Subdirectory name for images.
            annotations_dir: Subdirectory name for annotations.

        Raises:
            FileNotFoundError: If no COCO JSON file or no valid samples are found.
        """
        self.root = Path(root)
        images_path = self.root / images_dir
        annotations_path = self.root / annotations_dir

        json_files = sorted(annotations_path.glob("*.json"))
        if not json_files:
            json_files = sorted(self.root.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(
                f"No JSON annotation files found in {annotations_path} or {self.root}"
            )

        annotation_file = json_files[0]
        raw_samples, derived_classes = self._load_coco(
            images_path, annotation_file, default_class
        )

        self._classes = list(classes) if classes is not None else sorted(derived_classes)
        self._class_to_idx = {cls: idx for idx, cls in enumerate(self._classes)}

        self._samples: list[dict[str, Any]] = [
            s for s in raw_samples if s["image_label"] in self._class_to_idx
        ]
        for s in self._samples:
            s["image_label"] = self._class_to_idx[s["image_label"]]
            s["labels"] = [
                self._class_to_idx[lbl]
                for lbl in s["labels"]
                if lbl in self._class_to_idx
            ]
            s["boxes"] = [
                box
                for box, lbl in zip(s["boxes_raw"], s["labels_raw"])
                if lbl in self._class_to_idx
            ]

        if not self._samples:
            raise FileNotFoundError(f"No valid labeled samples found in {self.root}")

        if isinstance(transform, list):
            self.transform: A.Compose | None = self._build_transform(transform)
        else:
            self.transform = transform

        self._to_tensor = A.Compose(
            [ToTensorV2()],
            bbox_params=A.BboxParams(
                format="pascal_voc", label_fields=["labels"], min_visibility=0.1
            ),
        )

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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return ``(image_tensor, target)`` for the given index.

        ``target`` keys:
            - ``boxes``       — ``FloatTensor[N, 4]`` (x1, y1, x2, y2).
            - ``labels``      — ``LongTensor[N]``.
            - ``image_label`` — ``int``.
            - ``image_id``    — ``int``.
        """
        sample = self._samples[index]

        image = np.array(Image.open(sample["path"]).convert("RGB"))
        boxes = list(sample["boxes"])   # [[x1,y1,x2,y2], ...]
        labels = list(sample["labels"])  # [int, ...]

        if self.transform is not None:
            out = self.transform(image=image, bboxes=boxes, labels=labels)
            image = out["image"]
            boxes = list(out["bboxes"])
            labels = list(out["labels"])

        if not isinstance(image, torch.Tensor):
            out = self._to_tensor(image=image, bboxes=boxes, labels=labels)
            image = out["image"]
            boxes = list(out["bboxes"])
            labels = list(out["labels"])

        target: dict[str, Any] = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.long),
            "image_label": sample["image_label"],
            "image_id": sample["image_id"],
        }
        return image, target

    # ------------------------------------------------------------------
    # Collation
    # ------------------------------------------------------------------

    @staticmethod
    def collate_fn(
        batch: list[tuple[torch.Tensor, dict]],
    ) -> tuple[torch.Tensor, list[dict]]:
        """Collate a list of ``(image, target)`` samples into a batch.

        Images are stacked into a single tensor; targets are kept as a list
        because each image may have a different number of boxes.

        Args:
            batch: List of ``(image_tensor, target_dict)`` tuples.

        Returns:
            Tuple of ``(images, targets)`` where ``images`` is ``[B, C, H, W]``.
        """
        images, targets = zip(*batch)
        return torch.stack(images), list(targets)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_coco(
        images_path: Path,
        annotation_file: Path,
        default_class: str | None,
    ) -> tuple[list[dict[str, Any]], set[str]]:
        """Load samples from a single COCO-format JSON.

        Boxes are converted from COCO ``[x, y, w, h]`` to
        ``[x1, y1, x2, y2]`` (pascal_voc format).

        Args:
            images_path: Path to the images directory.
            annotation_file: Path to the COCO JSON file.
            default_class: Class for images with no annotations.

        Returns:
            Tuple of (samples list, set of all class names found).
        """
        data = json.loads(annotation_file.read_text())
        categories = {cat["id"]: cat["name"] for cat in data.get("categories", [])}

        img_to_anns: dict[int, list[dict]] = {}
        for ann in data.get("annotations", []):
            img_to_anns.setdefault(ann["image_id"], []).append(ann)

        samples: list[dict[str, Any]] = []
        all_classes: set[str] = set(categories.values())

        for img_info in data.get("images", []):
            img_path = images_path / img_info["file_name"]
            if not img_path.exists():
                continue

            anns = img_to_anns.get(img_info["id"], [])

            if not anns:
                if default_class is None:
                    continue
                all_classes.add(default_class)
                samples.append(
                    {
                        "path": img_path,
                        "image_id": img_info["id"],
                        "image_label": default_class,
                        "boxes_raw": [],
                        "labels_raw": [],
                        "boxes": [],
                        "labels": [],
                    }
                )
                continue

            boxes_xyxy = [
                [
                    ann["bbox"][0],
                    ann["bbox"][1],
                    ann["bbox"][0] + ann["bbox"][2],
                    ann["bbox"][1] + ann["bbox"][3],
                ]
                for ann in anns
            ]
            ann_labels = [categories[ann["category_id"]] for ann in anns]
            image_label = ann_labels[0]

            samples.append(
                {
                    "path": img_path,
                    "image_id": img_info["id"],
                    "image_label": image_label,
                    "boxes_raw": boxes_xyxy,
                    "labels_raw": ann_labels,
                    "boxes": boxes_xyxy,
                    "labels": ann_labels,
                }
            )

        return samples, all_classes

    @staticmethod
    def _build_transform(transform_list: list[dict]) -> A.Compose:
        """Build a bbox-aware ``A.Compose`` pipeline from a list of transform dicts.

        Args:
            transform_list: List of dicts with ``name`` and optional ``params``.

        Returns:
            ``A.Compose`` with ``BboxParams`` configured for pascal_voc format.
        """
        transforms = []
        for entry in transform_list:
            name = entry["name"]
            params = entry.get("params", {})
            if name == "ToTensorV2":
                transforms.append(ToTensorV2(**params))
            else:
                cls = getattr(A, name)
                transforms.append(cls(**params))

        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["labels"],
                min_visibility=0.1,
            ),
        )
