"""ImageFolder dataset — loads images from a directory tree organized by class."""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

from simpleml.registries import DATASETS

DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@DATASETS.register
class ImageFolderDataset(Dataset):
    """A dataset that reads images from a directory tree where each subdirectory
    represents a class.

    Expected layout::

        root/
            cat/
                img001.jpg
                img002.png
            dog/
                img003.jpg

    Images are loaded with PIL, converted to a NumPy array, passed through an
    optional Albumentations transform pipeline, and returned as
    ``(image_tensor, label_index)``.
    """

    def __init__(
        self,
        root: str,
        transform: A.Compose | None = None,
        extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
    ) -> None:
        self.root = Path(root)
        self.extensions = tuple(ext.lower() for ext in extensions)
        self.transform = transform

        self._classes = sorted(d.name for d in self.root.iterdir() if d.is_dir())
        if not self._classes:
            raise FileNotFoundError(f"No class subdirectories found in {self.root}")

        self._class_to_idx = {cls: idx for idx, cls in enumerate(self._classes)}

        self._samples: list[tuple[Path, int]] = []
        for cls_name in self._classes:
            cls_dir = self.root / cls_name
            label = self._class_to_idx[cls_name]
            for path in sorted(cls_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in self.extensions:
                    self._samples.append((path, label))

        if not self._samples:
            raise FileNotFoundError(
                f"No images with extensions {self.extensions} found in {self.root}"
            )

        self._to_tensor = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    @property
    def classes(self) -> list[str]:
        """Sorted list of class names."""
        return list(self._classes)

    @property
    def num_classes(self) -> int:
        """Number of classes discovered in the root directory."""
        return len(self._classes)

    @property
    def class_to_idx(self) -> dict[str, int]:
        """Mapping from class name to integer label."""
        return dict(self._class_to_idx)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Load and return ``(image_tensor, label)`` for the given index."""
        path, label = self._samples[index]

        image = Image.open(path).convert("RGB")
        image = np.array(image)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        if not isinstance(image, torch.Tensor):
            image = self._to_tensor(image=image)["image"]

        return image, label
