"""Shared utilities for detection metrics."""

from __future__ import annotations

import torch
from torch import Tensor


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Compute pairwise IoU between two sets of boxes.

    Args:
        boxes1: shape ``(N, 4)`` in ``[x1, y1, x2, y2]`` format.
        boxes2: shape ``(M, 4)`` in ``[x1, y1, x2, y2]`` format.

    Returns:
        IoU matrix of shape ``(N, M)``.
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union_area = area1[:, None] + area2[None, :] - inter_area

    return inter_area / union_area.clamp(min=1e-6)