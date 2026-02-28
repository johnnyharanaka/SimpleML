"""Supervised Contrastive Loss (Khosla et al., 2020)."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from simpleml.registries import LOSSES


@LOSSES.register
class SupConLoss(nn.Module):
    """Supervised contrastive loss.

    Pulls together embeddings that share the same label while pushing apart
    embeddings with different labels, using temperature-scaled cosine similarity.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, features: Tensor, labels: Tensor) -> Tensor:
        """Compute supervised contrastive loss.

        Args:
            features: Embeddings of shape ``(B, n_views, D)`` or ``(B, D)``.
                When 2-D, a single-view dimension is added automatically.
            labels: Integer class labels of shape ``(B,)``.

        Returns:
            Scalar loss.
        """
        if features.dim() == 2:
            features = features.unsqueeze(1)

        batch_size = features.shape[0]
        n_views = features.shape[1]

        features = nn.functional.normalize(features, dim=2)
        features_cat = features.reshape(batch_size * n_views, -1)

        labels_cat = labels.repeat(n_views)
        mask_pos = labels_cat.unsqueeze(0) == labels_cat.unsqueeze(1)

        similarity = features_cat @ features_cat.T / self.temperature

        n = batch_size * n_views
        self_mask = ~torch.eye(n, dtype=torch.bool, device=features.device)
        mask_pos = mask_pos & self_mask

        exp_sim = torch.exp(similarity) * self_mask.float()
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))

        pos_count = mask_pos.sum(dim=1).float()
        valid = pos_count > 0
        pos_log_sum = (log_prob * mask_pos.float()).sum(dim=1)
        mean_log_prob = pos_log_sum / pos_count.clamp(min=1)

        loss = -mean_log_prob[valid].mean()
        return loss
