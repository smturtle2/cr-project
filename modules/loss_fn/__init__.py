"""Loss 함수 모음.

- FrequencyLoss: FFT 기반 진폭+위상 loss
- HybridSpatialFrequencyLoss: L1 + FrequencyLoss 결합
- build_loss(): Trainer에 넘길 (prediction, batch) -> Tensor 형태의 callable 반환
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from .frequency_loss import FrequencyLoss
from .hybrid_loss import HybridSpatialFrequencyLoss


Batch = dict[str, Any]
LossFn = Callable[[torch.Tensor, Batch], torch.Tensor]


def build_loss(freq_weight: float = 0.1, phase_weight: float = 0.1) -> LossFn:
    """Trainer가 기대하는 (prediction, batch) -> Tensor 형태의 loss 함수 생성.

    기본값: L1 + 0.1 * (amplitude_l1 + 0.1 * phase_l1)
    """
    hybrid = HybridSpatialFrequencyLoss(freq_weight=freq_weight, phase_weight=phase_weight)

    def loss_fn(prediction: torch.Tensor, batch: Batch) -> torch.Tensor:
        return hybrid(prediction, batch)

    return loss_fn


__all__ = [
    "FrequencyLoss",
    "HybridSpatialFrequencyLoss",
    "build_loss",
    "LossFn",
]
