from __future__ import annotations

import torch

from .base import BaseDensityEstimator
from .cosine import CosineDensityEstimator
from .prior import OpticalRulePrior


class CosinePriorDensityEstimator(BaseDensityEstimator):
    """Blend the baseline cosine density with an optical-only prior."""

    def __init__(
        self,
        sar_channels: int = 2,
        optical_channels: int = 13,
        feat_dim: int = 32,
        prior_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.cosine = CosineDensityEstimator(
            sar_channels=sar_channels,
            optical_channels=optical_channels,
            feat_dim=feat_dim,
        )
        self.prior = OpticalRulePrior()
        self.prior_weight = float(prior_weight)

    def forward(self, sar: torch.Tensor, optical: torch.Tensor) -> torch.Tensor:
        cosine_density = self.cosine(sar, optical)
        prior_density = self.prior(optical)
        weight = self.prior_weight
        return ((1.0 - weight) * cosine_density + weight * prior_density).clamp(0.0, 1.0)
