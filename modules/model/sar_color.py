"""SAR 조건부 색상 예측 모듈.

구름이 두꺼운 영역에서는 Optical에 색상 정보가 거의 없으므로,
SAR 구조 특징으로부터 지표면 색상을 직접 예측하는 보조 경로를 제공한다.

- CloudDensityHead: SAR↔Optical 특징 유사도 기반 구름 밀도 추정 (0=맑음, 1=두꺼운 구름)
- SARColorPredictor: 디코더 특징(SAR+Optical 융합 완료)에서 13밴드 색상 직접 예측

최종 출력은 밀도에 따라 두 경로를 블렌딩:
  output = (1 - density) * (cloudy + residual) + density * sar_color
  - 맑은 영역: 기존 잔차 경로 신뢰 (Optical 색상 보존)
  - 구름 영역: SAR 기반 색상 예측 신뢰
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CloudDensityHead(nn.Module):
    """SAR-Optical 특징 유사도 기반 픽셀 단위 구름 밀도 추정.

    물리 원리:
      - SAR(레이더)은 구름 관통 → 지표면 특징이 항상 보임
      - Optical(빛)은 구름에 차단 → 구름 영역의 특징이 SAR과 달라짐
      - 두 특징의 유사도가 낮으면 → 구름이 있다는 신호

    입력: 인코더 stage 1 출력 (원본 해상도, 이미 학습된 특징)
    출력: (B, 1, H, W), 0=맑음, 1=두꺼운 구름
    """

    def __init__(self, feat_channels: int = 64) -> None:
        super().__init__()
        half = feat_channels // 2
        self.proj = nn.Conv2d(feat_channels, half, kernel_size=1)
        self.refine = nn.Sequential(
            nn.Conv2d(feat_channels * 2 + 1, half, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(half, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self, sar_feat: torch.Tensor, opt_feat: torch.Tensor
    ) -> torch.Tensor:
        sim = F.cosine_similarity(
            self.proj(sar_feat), self.proj(opt_feat), dim=1, eps=1e-6
        )
        # 유사도가 높으면 맑음, 낮으면 구름 → 반전해서 density로 사용
        dissim = (1.0 - sim).unsqueeze(1)  # (B, 1, H, W)
        density = self.refine(torch.cat([sar_feat, opt_feat, dissim], dim=1))
        return density


class SARColorPredictor(nn.Module):
    """디코더 특징으로부터 13밴드 지표면 색상을 직접 예측.

    d1 특징에는 이미 CrossFrequencyFusion을 통해 SAR의 구조 정보가
    깊이 융합되어 있으므로, 이로부터 색상을 예측하면
    "이 SAR 구조 패턴일 때 어떤 색상이 나올까?"를 학습하게 된다.

    입력: 디코더 stage 1 최종 출력 (B, base_channels, H, W)
    출력: (B, opt_channels, H, W), 예측된 지표면 색상
    """

    def __init__(self, in_channels: int = 64, out_channels: int = 13) -> None:
        super().__init__()
        mid = in_channels // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels, mid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid, out_channels, kernel_size=1),
        )

    def forward(self, decoder_feat: torch.Tensor) -> torch.Tensor:
        return self.net(decoder_feat)
