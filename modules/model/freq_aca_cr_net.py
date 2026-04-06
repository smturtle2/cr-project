"""ACA-CRNet baseline + AFR-CR의 Frequency Domain Fusion 모듈을 결합한 모델."""

from __future__ import annotations

import torch
from torch import nn

from .blocks import ChannelAttention, ConvBlock
from .freq_blocks import CrossFrequencyFusion
from .sar_color import CloudDensityHead, SARColorPredictor


class FreqACACRNet(nn.Module):
    """Baseline: ACA-CRNet 스타일 (SAR-Optical 듀얼 브랜치 + Channel Attention)
    + AFR-CR의 주파수 도메인 교차 재구성 (CrossFrequencyFusion).

    입력:
      - sar:    (B, 2, H, W)  Sentinel-1 VV/VH
      - cloudy: (B, 13, H, W) Sentinel-2 구름 있는 광학

    출력:
      - (B, 13, H, W) 구름 제거된 광학 (cloudy + residual 형태)

    구조:
      - Encoder: 각 모달리티(sar, optical) 독립 2-stage 인코더
      - Bottleneck: CrossFrequencyFusion으로 저/고주파 교차 재구성
      - Decoder: Skip connection에도 CrossFrequencyFusion 적용
      - Head: 13채널 residual 출력 -> cloudy에 더해 최종 결과 생성
      - SAR Color: 구름 두꺼운 영역에서 SAR 구조 기반 색상 직접 예측
      - Density Blend: 구름 밀도에 따라 잔차 경로와 SAR 색상 경로를 픽셀별 블렌딩
    """

    def __init__(
        self,
        sar_channels: int = 2,
        opt_channels: int = 13,
        base_channels: int = 64,
    ) -> None:
        super().__init__()

        # 각 모달리티용 초기 임베딩
        self.sar_embed = ConvBlock(sar_channels, base_channels)
        self.opt_embed = ConvBlock(opt_channels, base_channels)

        # Encoder Stage 1
        self.enc1_sar = ConvBlock(base_channels, base_channels)
        self.enc1_opt = ConvBlock(base_channels, base_channels)
        self.down1 = nn.AvgPool2d(2)

        # Encoder Stage 2
        self.enc2_sar = ConvBlock(base_channels, base_channels * 2)
        self.enc2_opt = ConvBlock(base_channels, base_channels * 2)
        self.down2 = nn.AvgPool2d(2)

        # Bottleneck: 주파수 도메인 교차 재구성 (AFR-CR 핵심)
        self.freq_fusion_bottleneck = CrossFrequencyFusion(base_channels * 2)
        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 2)
        self.ca_bottleneck = ChannelAttention(base_channels * 2)

        # Decoder Stage 2 (skip에도 주파수 융합 적용)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.freq_fusion_dec2 = CrossFrequencyFusion(base_channels * 2)
        self.dec2 = ConvBlock(base_channels * 2 + base_channels * 2, base_channels)
        self.ca_dec2 = ChannelAttention(base_channels)

        # Decoder Stage 1
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.freq_fusion_dec1 = CrossFrequencyFusion(base_channels)
        self.dec1 = ConvBlock(base_channels + base_channels, base_channels)
        self.ca_dec1 = ChannelAttention(base_channels)

        # 최종 복원 head: 13 채널 residual 출력
        self.head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, opt_channels, kernel_size=1),
        )

        # SAR 조건부 색상 예측 (구름 두꺼운 영역용)
        self.cloud_density = CloudDensityHead(base_channels)
        self.sar_color = SARColorPredictor(base_channels, opt_channels)

    def forward(self, sar: torch.Tensor, cloudy: torch.Tensor) -> torch.Tensor:
        # 초기 임베딩
        s0 = self.sar_embed(sar)
        o0 = self.opt_embed(cloudy)

        # Encoder Stage 1
        s1 = self.enc1_sar(s0)
        o1 = self.enc1_opt(o0)
        s1_d = self.down1(s1)
        o1_d = self.down1(o1)

        # Encoder Stage 2
        s2 = self.enc2_sar(s1_d)
        o2 = self.enc2_opt(o1_d)
        s2_d = self.down2(s2)
        o2_d = self.down2(o2)

        # Bottleneck: 주파수 도메인 교차 재구성
        fused = self.freq_fusion_bottleneck(o2_d, s2_d)
        fused = self.bottleneck(fused + o2_d)  # residual 연결로 안정성 향상
        fused = self.ca_bottleneck(fused)

        # Decoder Stage 2
        fused_up = self.up2(fused)
        skip2 = self.freq_fusion_dec2(o2, s2)  # skip 연결에도 주파수 융합
        d2 = self.dec2(torch.cat([fused_up, skip2], dim=1))
        d2 = self.ca_dec2(d2)

        # Decoder Stage 1
        d2_up = self.up1(d2)
        skip1 = self.freq_fusion_dec1(o1, s1)
        d1 = self.dec1(torch.cat([d2_up, skip1], dim=1))
        d1 = self.ca_dec1(d1)

        # 경로 1: 잔차 복원 (맑은/얇은 구름 영역에 효과적)
        residual = self.head(d1)
        residual_output = cloudy + residual

        # 경로 2: SAR 기반 색상 직접 예측 (두꺼운 구름 영역에 효과적)
        sar_color = self.sar_color(d1)

        # 구름 밀도 추정 → 두 경로를 픽셀별로 블렌딩
        # density ≈ 0 (맑음): residual_output 신뢰
        # density ≈ 1 (구름): sar_color 신뢰
        density = self.cloud_density(s1, o1)
        return (1.0 - density) * residual_output + density * sar_color
