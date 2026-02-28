"""CorLoc metric registered for config-driven use."""

from __future__ import annotations

from torch import Tensor

from simpleml.metrics._detection_utils import box_iou
from simpleml.metrics.base import Metric
from simpleml.registries import METRICS


@METRICS.register
class CorLoc(Metric):
    """Correct Localization (CorLoc) for object detection.

    Measures the fraction of images for which the highest-scoring predicted
    bounding box has IoU ≥ ``iou_threshold`` with at least one ground-truth box.

    Each call to ``update`` processes **one image**:

    - ``preds``: shape ``(N, 5)`` — detected boxes in ``[x1, y1, x2, y2, score]`` format.
    - ``targets``: shape ``(M, 4)`` — ground-truth boxes in ``[x1, y1, x2, y2]`` format.

    An image with zero predictions or zero ground-truth boxes counts as a miss.

    Args:
        iou_threshold: Minimum IoU for a detection to be considered correct.
            Defaults to ``0.5``.
    """

    def __init__(self, iou_threshold: float = 0.5) -> None:
        super().__init__()
        self.iou_threshold = iou_threshold
        self._correct: list[bool] = []

    def update(self, preds: Tensor, targets: Tensor) -> None:
        """Evaluate one image and accumulate the localization result.

        Args:
            preds: shape ``(N, 5)`` with columns ``[x1, y1, x2, y2, score]``.
            targets: shape ``(M, 4)`` with columns ``[x1, y1, x2, y2]``.
        """
        if preds.shape[0] == 0 or targets.shape[0] == 0:
            self._correct.append(False)
            return
        top_box = preds[preds[:, 4].argmax(), :4].unsqueeze(0)
        ious = box_iou(top_box, targets)[0]
        self._correct.append(bool(ious.max().item() >= self.iou_threshold))

    def compute(self) -> float:
        """Return CorLoc as the fraction of correctly localized images."""
        if not self._correct:
            raise RuntimeError(
                f"{type(self).__name__}.compute() called before any update()."
            )
        return float(sum(self._correct) / len(self._correct))

    def reset(self) -> None:
        """Clear accumulated localization results."""
        super().reset()
        self._correct.clear()

    def __call__(self, preds: Tensor, targets: Tensor) -> float:  # type: ignore[override]
        """Reset, update with the given image, and compute CorLoc."""
        self.reset()
        self.update(preds, targets)
        return self.compute()
