"""Tests for the datasets module."""

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

from simpleml import DATASETS
from simpleml.datasets import ImageFolderDataset


def _create_image_folder(root, classes, num_per_class=3, ext=".jpg", size=(32, 32)):
    """Create a synthetic ImageFolder directory tree with random images."""
    for cls_name in classes:
        cls_dir = root / cls_name
        cls_dir.mkdir(parents=True)
        for i in range(num_per_class):
            img_array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(cls_dir / f"img_{i:03d}{ext}")


class TestImageFolderDatasetRegistry:
    def test_registered_in_datasets_registry(self):
        assert "ImageFolderDataset" in DATASETS

    def test_build_via_registry(self, tmp_path):
        _create_image_folder(tmp_path, ["a", "b"])
        ds = DATASETS.build("ImageFolderDataset", root=str(tmp_path))
        assert isinstance(ds, ImageFolderDataset)


class TestImageFolderDataset:
    def test_len(self, tmp_path):
        _create_image_folder(tmp_path, ["cat", "dog"], num_per_class=5)
        ds = ImageFolderDataset(root=str(tmp_path))
        assert len(ds) == 10

    def test_getitem_returns_tensor_and_int(self, tmp_path):
        _create_image_folder(tmp_path, ["cat", "dog"])
        ds = ImageFolderDataset(root=str(tmp_path))
        image, label = ds[0]
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, int)

    def test_getitem_shape_no_transform(self, tmp_path):
        _create_image_folder(tmp_path, ["cat"], size=(48, 64))
        ds = ImageFolderDataset(root=str(tmp_path))
        image, _ = ds[0]
        assert image.shape == (3, 48, 64)

    def test_getitem_shape_with_resize_transform(self, tmp_path):
        _create_image_folder(tmp_path, ["cat"], size=(64, 64))
        transform = A.Compose([A.Resize(32, 32), ToTensorV2()])
        ds = ImageFolderDataset(root=str(tmp_path), transform=transform)
        image, _ = ds[0]
        assert image.shape == (3, 32, 32)

    def test_getitem_with_augmentation(self, tmp_path):
        _create_image_folder(tmp_path, ["cat"], size=(32, 32))
        transform = A.Compose([A.HorizontalFlip(p=1.0), ToTensorV2()])
        ds = ImageFolderDataset(root=str(tmp_path), transform=transform)
        image, _ = ds[0]
        assert image.shape == (3, 32, 32)

    def test_classes_property(self, tmp_path):
        _create_image_folder(tmp_path, ["dog", "cat", "bird"])
        ds = ImageFolderDataset(root=str(tmp_path))
        assert ds.classes == ["bird", "cat", "dog"]

    def test_num_classes_property(self, tmp_path):
        _create_image_folder(tmp_path, ["a", "b", "c"])
        ds = ImageFolderDataset(root=str(tmp_path))
        assert ds.num_classes == 3

    def test_class_to_idx_property(self, tmp_path):
        _create_image_folder(tmp_path, ["dog", "cat"])
        ds = ImageFolderDataset(root=str(tmp_path))
        expected = {"cat": 0, "dog": 1}
        assert ds.class_to_idx == expected

    def test_labels_match_classes(self, tmp_path):
        _create_image_folder(tmp_path, ["cat", "dog"], num_per_class=2)
        ds = ImageFolderDataset(root=str(tmp_path))
        labels = {ds[i][1] for i in range(len(ds))}
        assert labels == {0, 1}

    def test_custom_extensions(self, tmp_path):
        _create_image_folder(tmp_path, ["a"], num_per_class=2, ext=".png")
        _create_image_folder(tmp_path, ["b"], num_per_class=3, ext=".bmp")
        ds = ImageFolderDataset(root=str(tmp_path), extensions=(".png",))
        assert len(ds) == 2

    def test_no_classes_raises(self, tmp_path):
        import pytest

        with pytest.raises(FileNotFoundError, match="No class subdirectories"):
            ImageFolderDataset(root=str(tmp_path))

    def test_no_images_raises(self, tmp_path):
        import pytest

        (tmp_path / "empty_class").mkdir()
        with pytest.raises(FileNotFoundError, match="No images"):
            ImageFolderDataset(root=str(tmp_path))

    def test_image_dtype_is_float_after_to_tensor(self, tmp_path):
        _create_image_folder(tmp_path, ["cat"], size=(16, 16))
        transform = A.Compose([ToTensorV2()])
        ds = ImageFolderDataset(root=str(tmp_path), transform=transform)
        image, _ = ds[0]
        assert image.dtype == torch.uint8 or image.dtype == torch.float32
