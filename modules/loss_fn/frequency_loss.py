"""주파수 도메인 loss.

AFR-CR이 사용하는 주파수 도메인 loss의 핵심 아이디어:
  - 진폭(amplitude) loss: 구름 성분 억제에 직접적으로 기여
    (구름은 저주파 진폭에 집중되므로, 진폭을 맞추면 구름 성분도 맞춰진다)
  - 위상(phase) loss: 구조/경계/물체 위치 보존
    (위상은 영상의 구조 정보를 담고 있으므로 지표면 디테일 보존에 중요)
"""

from __future__ import annotations

import torch
from torch import nn


class FrequencyLoss(nn.Module):
    """FFT 기반 주파수 도메인 L1 loss (진폭 + 위상)."""

    def __init__(self, phase_weight: float = 0.1) -> None:
        super().__init__()
        self.l1 = nn.L1Loss()
        self.phase_weight = phase_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 두 영상을 주파수 도메인으로 변환
        pred_freq = torch.fft.fft2(pred, norm="ortho")
        target_freq = torch.fft.fft2(target, norm="ortho")

        # 진폭과 위상으로 분리
        pred_amp = torch.abs(pred_freq)
        pred_phase = torch.angle(pred_freq)
        target_amp = torch.abs(target_freq)
        target_phase = torch.angle(target_freq)

        # 진폭 loss
        amp_loss = self.l1(pred_amp, target_amp)

        # 위상 loss: 각도의 주기성을 고려해 cos/sin 공간에서 비교
        phase_loss = self.l1(torch.cos(pred_phase), torch.cos(target_phase)) + \
                     self.l1(torch.sin(pred_phase), torch.sin(target_phase))

        return amp_loss + self.phase_weight * phase_loss
