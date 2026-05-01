from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseDensityEstimator(nn.Module, ABC):
    """Common interface for pluggable density estimators."""

    @abstractmethod
    def forward(self, sar: torch.Tensor, optical: torch.Tensor) -> torch.Tensor:
        """Return a `(B, 1, H, W)` density map in `[0, 1]`."""

