from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseGateEstimator(nn.Module, ABC):
    """Common interface for SAR injection gate estimators."""

    @abstractmethod
    def forward(self, sar: torch.Tensor, optical: torch.Tensor) -> torch.Tensor:
        """Return a gate map with shape (B, 1, H, W) and values in [0, 1]."""
