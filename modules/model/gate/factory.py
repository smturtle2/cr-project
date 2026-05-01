from __future__ import annotations

from .base import BaseGateEstimator
from .cosine import CosineGateEstimator
from .cosine_prior import CosinePriorGateEstimator
from .prior import OpticalRulePrior


def build_gate_estimator(
    mode: str,
    *,
    sar_channels: int = 2,
    optical_channels: int = 13,
    feat_dim: int = 32,
    prior_weight: float = 0.5,
) -> BaseGateEstimator | None:
    if mode in ("mask", "none", None):
        return None
    if mode == "cosine":
        return CosineGateEstimator(
            sar_channels=sar_channels,
            optical_channels=optical_channels,
            feat_dim=feat_dim,
        )
    if mode == "cosine_prior":
        return CosinePriorGateEstimator(
            sar_channels=sar_channels,
            optical_channels=optical_channels,
            feat_dim=feat_dim,
            prior_weight=prior_weight,
        )
    if mode == "prior":
        return OpticalRulePrior()
    raise ValueError(f"unsupported gate mode: {mode}")
