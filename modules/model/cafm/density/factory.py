from __future__ import annotations

from .base import BaseDensityEstimator
from .cosine import CosineDensityEstimator
from .cosine_prior import CosinePriorDensityEstimator


def build_density_estimator(
    mode: str,
    *,
    sar_channels: int = 2,
    optical_channels: int = 13,
    feat_dim: int = 32,
) -> BaseDensityEstimator:
    if mode == "cosine":
        return CosineDensityEstimator(
            sar_channels=sar_channels,
            optical_channels=optical_channels,
            feat_dim=feat_dim,
        )
    if mode == "cosine_prior":
        return CosinePriorDensityEstimator(
            sar_channels=sar_channels,
            optical_channels=optical_channels,
            feat_dim=feat_dim,
        )
    raise ValueError(f"unsupported density mode: {mode}")
