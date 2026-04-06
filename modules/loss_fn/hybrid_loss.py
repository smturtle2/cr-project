"""공간 도메인 + 주파수 도메인을 결합한 하이브리드 loss."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .frequency_loss import FrequencyLoss


Batch = dict[str, Any]


class HybridSpatialFrequencyLoss(nn.Module):
    """공간 도메인 L1 + 주파수 도메인 loss를 가중합.

    - spatial (L1): 픽셀 단위 복원 정확도
    - frequency: 구름 성분 억제 + 구조 보존
    """

    def __init__(self, freq_weight: float = 0.1, phase_weight: float = 0.1) -> None:
        super().__init__()
        self.l1 = nn.L1Loss()
        self.freq_loss = FrequencyLoss(phase_weight=phase_weight)
        self.freq_weight = freq_weight

    def forward(self, prediction: torch.Tensor, batch: Batch) -> torch.Tensor:
        target = batch["target"]
        spatial = self.l1(prediction, target)
        frequency = self.freq_loss(prediction, target)
        return spatial + self.freq_weight * frequency
