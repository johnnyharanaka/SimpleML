"""Mean Average Precision for object detection, registered for config-driven use."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch import Tensor

from simpleml.metrics._detection_utils import box_iou
from simpleml.metrics.base import Metric
from simpleml.registries import METRICS


def _compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute AP using all-point interpolation (VOC 2010+ style)."""
    recall = np.concatenate([[0.0], recall, [1.0]])
    precision = np.concatenate([[1.0], precision, [0.0]])
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    idx = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[idx + 1] - recall[idx]) * precision[idx + 1]))


@METRICS.register
class MeanAveragePrecision(Metric):
    """Mean Average Precision (mAP) for object detection.

    Accumulates per-image detection predictions and ground truths, then
    computes AP per class using IoU-based matching and averages across all
    classes that appear in the ground truth.

    Each call to ``update`` receives one batch of images:

    - ``preds``: list of per-image dicts with keys
      ``"boxes"`` (Tensor N×4), ``"scores"`` (Tensor N,), ``"labels"`` (Tensor N,).
    - ``targets``: list of per-image dicts with keys
      ``"boxes"`` (Tensor M×4), ``"labels"`` (Tensor M,).

    All boxes must be in ``[x1, y1, x2, y2]`` format.

    Args:
        iou_threshold: Minimum IoU to count a detection as a true positive.
            Defaults to ``0.5`` (mAP@0.5).

    Example::

        metric = METRICS.build("MeanAveragePrecision")
        metric.update(
            preds=[{"boxes": pred_boxes, "scores": scores, "labels": labels}],
            targets=[{"boxes": gt_boxes, "labels": gt_labels}],
        )
        map_score = metric.compute()
    """

    def __init__(self, iou_threshold: float = 0.5) -> None:
        super().__init__()
        self.iou_threshold = iou_threshold
        self._image_preds: list[dict] = []
        self._image_targets: list[dict] = []

    def update(self, preds: list[dict], targets: list[dict]) -> None:  # type: ignore[override]
        """Accumulate detection predictions and ground truths for a batch.

        Args:
            preds: List of per-image dicts with keys ``"boxes"``, ``"scores"``,
                and ``"labels"``.
            targets: List of per-image dicts with keys ``"boxes"`` and ``"labels"``.
        """
        self._image_preds.extend(preds)
        self._image_targets.extend(targets)

    def compute(self) -> float:
        """Return mAP averaged over all ground-truth classes."""
        if not self._image_preds:
            raise RuntimeError(
                f"{type(self).__name__}.compute() called before any update()."
            )

        gt_by_class: dict[int, dict[int, Tensor]] = defaultdict(dict)
        det_by_class: dict[int, list[tuple[float, int, int]]] = defaultdict(list)

        for img_id, target in enumerate(self._image_targets):
            boxes = target["boxes"]
            labels = target["labels"]
            for cls in labels.unique().tolist():
                cls = int(cls)
                gt_by_class[cls][img_id] = boxes[labels == cls]

        for img_id, pred in enumerate(self._image_preds):
            boxes = pred["boxes"]
            scores = pred["scores"]
            labels = pred["labels"]
            for i in range(len(scores)):
                cls = int(labels[i].item())
                det_by_class[cls].append((float(scores[i].item()), img_id, i))

        all_classes = set(gt_by_class.keys())
        if not all_classes:
            return 0.0

        aps: list[float] = []
        for cls in all_classes:
            detections = sorted(det_by_class[cls], key=lambda x: -x[0])
            gt_cls = gt_by_class[cls]
            n_gt = sum(v.shape[0] for v in gt_cls.values())
            matched: dict[int, set[int]] = defaultdict(set)
            tp = np.zeros(len(detections))
            fp = np.zeros(len(detections))

            for det_idx, (_, img_id, box_idx) in enumerate(detections):
                pred_box = self._image_preds[img_id]["boxes"][box_idx].unsqueeze(0)
                gt_boxes = gt_cls.get(img_id)
                if gt_boxes is None or gt_boxes.shape[0] == 0:
                    fp[det_idx] = 1
                    continue
                ious = box_iou(pred_box, gt_boxes)[0]
                best_iou, best_gt = ious.max(0)
                best_gt = int(best_gt.item())
                if float(best_iou.item()) >= self.iou_threshold and best_gt not in matched[img_id]:
                    tp[det_idx] = 1
                    matched[img_id].add(best_gt)
                else:
                    fp[det_idx] = 1

            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            recall = cum_tp / (n_gt + 1e-9)
            precision = cum_tp / (cum_tp + cum_fp + 1e-9)
            aps.append(_compute_ap(recall, precision))

        return float(np.mean(aps))

    def reset(self) -> None:
        """Clear accumulated detection predictions and ground truths."""
        super().reset()
        self._image_preds.clear()
        self._image_targets.clear()

    def __call__(self, preds: list[dict], targets: list[dict]) -> float:  # type: ignore[override]
        """Reset, update with the given batch, and compute mAP."""
        self.reset()
        self.update(preds, targets)
        return self.compute()
