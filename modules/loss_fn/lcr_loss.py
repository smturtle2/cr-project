from __future__ import annotations

import torch
from torch import nn

from modules.metrics import SAM, SSIM


class LCRLoss(nn.Module):
    def __init__(
        self,
        *,
        l1_weight: float = 1.0,
        ssim_weight: float = 0.5,
        sam_weight: float = 0.01,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.sam_weight = sam_weight
        self.ssim = SSIM()
        self.sam = SAM()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = torch.mean(torch.abs(prediction - target))
        ssim_loss = 1.0 - self.ssim(prediction, target)
        sam_loss = self.sam(prediction, target)
        return (
            self.l1_weight * l1
            + self.ssim_weight * ssim_loss
            + self.sam_weight * sam_loss
        )
