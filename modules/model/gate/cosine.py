from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .base import BaseGateEstimator


class CosineGateEstimator(BaseGateEstimator):
    """SAR-optical cosine similarity gate followed by a light refiner."""

    def __init__(
        self,
        sar_channels: int = 2,
        optical_channels: int = 13,
        feat_dim: int = 32,
    ) -> None:
        super().__init__()
        self.sar_encoder = nn.Sequential(
            nn.Conv2d(sar_channels, feat_dim, kernel_size=1, bias=True),
            nn.GELU(),
        )
        self.optical_encoder = nn.Sequential(
            nn.Conv2d(optical_channels, feat_dim, kernel_size=1, bias=True),
            nn.GELU(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(2 * feat_dim + 1, feat_dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(feat_dim, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def encode(
        self,
        sar: torch.Tensor,
        optical: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sar_feat = self.sar_encoder(sar)
        optical_feat = self.optical_encoder(optical)
        similarity = F.cosine_similarity(
            sar_feat,
            optical_feat,
            dim=1,
            eps=1e-6,
        ).unsqueeze(1)
        return sar_feat, optical_feat, similarity

    def forward(self, sar: torch.Tensor, optical: torch.Tensor) -> torch.Tensor:
        sar_feat, optical_feat, similarity = self.encode(sar, optical)
        features = torch.cat([sar_feat, optical_feat, similarity], dim=1)
        return self.refine(features)
