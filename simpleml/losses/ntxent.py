"""NT-Xent (Normalized Temperature-scaled Cross Entropy) loss for SimCLR."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from simpleml.registries import LOSSES


@LOSSES.register
class NTXentLoss(nn.Module):
    """NT-Xent / InfoNCE contrastive loss.

    Given two augmented views of the same batch, treats the paired view as
    the positive and all other samples as negatives.
    """

    def __init__(self, temperature: float = 0.5) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: Tensor, z_j: Tensor) -> Tensor:
        """Compute NT-Xent loss.

        Args:
            z_i: Embeddings from the first view, shape ``(B, D)``.
            z_j: Embeddings from the second view, shape ``(B, D)``.

        Returns:
            Scalar loss.
        """
        batch_size = z_i.shape[0]

        z_i = nn.functional.normalize(z_i, dim=1)
        z_j = nn.functional.normalize(z_j, dim=1)

        z = torch.cat([z_i, z_j], dim=0)
        similarity = z @ z.T / self.temperature

        n = 2 * batch_size
        self_mask = ~torch.eye(n, dtype=torch.bool, device=z.device)
        similarity = similarity.masked_fill(~self_mask, float("-inf"))

        labels = torch.cat(
            [
                torch.arange(batch_size, 2 * batch_size, device=z.device),
                torch.arange(0, batch_size, device=z.device),
            ],
        )

        loss = nn.functional.cross_entropy(similarity, labels)
        return loss
