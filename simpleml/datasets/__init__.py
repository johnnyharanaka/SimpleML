"""Dataset wrappers registered for config-driven use."""

from simpleml.datasets.coco_classification import COCOClassificationDataset
from simpleml.datasets.coco_detection import COCODetectionDataset
from simpleml.datasets.image_folder import ImageFolderDataset

__all__ = ["COCOClassificationDataset", "COCODetectionDataset", "ImageFolderDataset"]
