from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .base import BaseGateEstimator


def _safe_rescale(x: torch.Tensor, low: float, high: float) -> torch.Tensor:
    scale = high - low
    if abs(scale) < 1e-6:
        scale = 1e-6
    return ((x - low) / scale).clamp(0.0, 1.0)


def _normalized_difference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    denom = (a + b).clamp_min(1e-6)
    return (a - b) / denom


class OpticalRulePrior(BaseGateEstimator):
    """Sentinel-2 optical-only rule-based cloud-likeness gate prior."""

    def forward(self, sar: torch.Tensor, optical: torch.Tensor) -> torch.Tensor:
        del sar
        aerosol = optical[:, 0:1]
        blue = optical[:, 1:2]
        green = optical[:, 2:3]
        red = optical[:, 3:4]
        nir = optical[:, 7:8]
        cirrus = optical[:, 10:11]
        swir1 = optical[:, 11:12]

        score = torch.ones_like(blue)
        score = torch.minimum(score, _safe_rescale(blue, 0.1, 0.5))
        score = torch.minimum(score, _safe_rescale(aerosol, 0.1, 0.3))
        score = torch.minimum(score, _safe_rescale(aerosol + cirrus, 0.4, 0.9))
        score = torch.minimum(score, _safe_rescale(red + green + blue, 0.2, 0.8))

        ndmi = _normalized_difference(nir, swir1)
        score = torch.minimum(score, _safe_rescale(ndmi, -0.1, 0.1))

        ndsi = _normalized_difference(green, swir1)
        score = torch.minimum(score, _safe_rescale(ndsi, 0.8, 0.6))

        score = F.max_pool2d(score, kernel_size=5, stride=1, padding=2)
        score = F.avg_pool2d(score, kernel_size=7, stride=1, padding=3)
        return score.clamp(0.0, 1.0)
