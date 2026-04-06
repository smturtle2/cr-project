"""Cloud removal 모델 모음.

- FreqACACRNet: ACA-CRNet baseline + AFR-CR Frequency Fusion + SAR 색상 예측
- CrossFrequencyFusion: SAR-Optical 주파수 도메인 교차 재구성 모듈
- FrequencyDecomposition / FrequencyReconstruction: FFT 기반 저/고주파 분해/합성
- CloudDensityHead / SARColorPredictor: 구름 밀도 기반 SAR 조건부 색상 복원
"""

from __future__ import annotations

from torch import nn

from .blocks import ChannelAttention, ConvBlock
from .freq_aca_cr_net import FreqACACRNet
from .freq_blocks import (
    CrossFrequencyFusion,
    FrequencyDecomposition,
    FrequencyReconstruction,
)
from .sar_color import CloudDensityHead, SARColorPredictor


def build_model(
    sar_channels: int = 2,
    opt_channels: int = 13,
    base_channels: int = 64,
) -> nn.Module:
    """SEN12MS-CR용 기본 모델(FreqACACRNet) 빌더."""
    return FreqACACRNet(
        sar_channels=sar_channels,
        opt_channels=opt_channels,
        base_channels=base_channels,
    )


__all__ = [
    "FreqACACRNet",
    "CrossFrequencyFusion",
    "FrequencyDecomposition",
    "FrequencyReconstruction",
    "ConvBlock",
    "ChannelAttention",
    "CloudDensityHead",
    "SARColorPredictor",
    "build_model",
]
