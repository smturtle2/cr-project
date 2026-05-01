from __future__ import annotations

import torch

from .base import BaseGateEstimator
from .cosine import CosineGateEstimator
from .prior import OpticalRulePrior


class CosinePriorGateEstimator(BaseGateEstimator):
    """Blend learned cosine gate with a fixed optical-only prior."""

    def __init__(
        self,
        sar_channels: int = 2,
        optical_channels: int = 13,
        feat_dim: int = 32,
        prior_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.cosine = CosineGateEstimator(
            sar_channels=sar_channels,
            optical_channels=optical_channels,
            feat_dim=feat_dim,
        )
        self.prior = OpticalRulePrior()
        self.prior_weight = float(prior_weight)

    def forward(self, sar: torch.Tensor, optical: torch.Tensor) -> torch.Tensor:
        cosine_gate = self.cosine(sar, optical)
        prior_gate = self.prior(sar, optical)
        weight = self.prior_weight
        return ((1.0 - weight) * cosine_gate + weight * prior_gate).clamp(0.0, 1.0)
