# -*- coding: utf-8 -*-
"""
Cloud-Adaptive Feature Modulation (CAFM) 모듈
============================================
[신규 파일] ACA-CRNet에 플러그인으로 장착되는 구름 적응 변조 모듈.

핵심 아이디어:
    SAR-Optical 코사인 유사도로 구름 밀도를 추정하고,
    밀도에 따라 네트워크 feature를 AdaIN-style로 변조한다.
    맑은 영역은 보존, 구름 영역은 강하게 재구성.

참고 논문 및 코드 출처:
    [1] MWFormer (IEEE TIP, 2024) — FiLM 변조: X' = γ·X + β
        코드: https://github.com/taco-group/MWFormer/blob/main/model/EncDec.py
        FilmBlock 클래스의 y_weight(γ), y_bias(β) 생성 패턴 참조

    [2] EMRDM (CVPR 2025) — Scale-shift norm + zero_module 초기화
        코드: https://github.com/Ly403/EMRDM/blob/main/sgm/modules/diffusionmodules/openaimodel.py
        ResBlock의 (1 + scale) * h + shift 패턴 및 zero_module 참조

    [3] SMDCNet (ISPRS 2025) — SAR-Optical feature 유사도 기반 구름 감지
        "cloud-covered regions weaken optical-SAR feature correlations"
        코사인 유사도를 구름 밀도 프록시로 사용하는 물리적 근거 제공

    [4] DFPIR (CVPR 2025) — 열화 인지 적응 처리
        열화 조건에서 feature를 조건부 변조하는 전략의 효과 검증 (ablation +0.90dB)

    [5] StyleGAN2 — AdaIN (Adaptive Instance Normalization)
        feat * (1 + γ) + β 형태의 조건부 변조 원형
"""

import torch
import torch.nn as nn

from .density import CosineDensityEstimator as CloudDensityEstimator


# ============================================================================
# 유틸리티: zero_module
# 출처: EMRDM — sgm/modules/diffusionmodules/util.py
# 모듈의 모든 파라미터를 0으로 초기화하여 잔차 학습 시 항등 함수로 시작
# ============================================================================
def zero_module(module: nn.Module) -> nn.Module:
    """모듈의 모든 파라미터를 0으로 초기화하여 반환한다.
    학습 초기에 변조 출력이 0이 되어 항등 함수(identity)로 동작하게 한다.

    참조: EMRDM (CVPR 2025)
        https://github.com/Ly403/EMRDM/blob/main/sgm/modules/diffusionmodules/util.py
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


# ============================================================================
# Part B: 적응적 Feature 변조 (Adaptive Feature Modulation)
# ============================================================================
class AdaptiveFeatureModulator(nn.Module):
    """AdaIN-style 채널별 Feature 변조기. (GAP 한계 극복 및 공간 변조 적용)

    수식: output = feature * (1 + γ) + β
        - γ(scale), β(shift)는 밀도 조건에서 생성
        - γ=0, β=0일 때 항등 함수 (맑은 영역 보존)
        - γ>0, β>0일 때 feature 강하게 변형 (구름 영역 재구성)

    Args:
        feature_channels: 변조할 feature의 채널 수 (기본 256 = ACA-CRNet feature_sizes)
    """

    def __init__(self, feature_channels: int = 256):
        super().__init__()
        C = feature_channels

        # --------------------------------------------------------------------
        # [AS-IS] 기존 코드 (GAP 기반 전역 변조) - 주석 처리
        # --------------------------------------------------------------------
        # # 조건 인코딩: GAP된 feature(C) + GAP된 density(1) = C+1 차원
        # self.scale_net = nn.Sequential(
        #     nn.Linear(C + 1, C // 4),
        #     nn.GELU(),
        #     zero_module(nn.Linear(C // 4, C)),  # zero-init: 초기 γ=0
        # )
        # self.shift_net = nn.Sequential(
        #     nn.Linear(C + 1, C // 4),
        #     nn.GELU(),
        #     zero_module(nn.Linear(C // 4, C)),  # zero-init: 초기 β=0
        # )

        # --------------------------------------------------------------------
        # [TO-BE] 신규 코드 (공간 가변 변조 - Spatial Adaptive Modulation)
        # --------------------------------------------------------------------
        hidden_dim = max(32, C // 4)

        # Density Map (1ch)을 입력받아 변조에 필요한 특징 추출 (공간 HxW 유지)
        self.shared_conv = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1),
            nn.GELU()
        )

        # 픽셀 단위 스케일(Gamma) 맵과 시프트(Beta) 맵 생성용 Conv 레이어
        self.conv_gamma = nn.Conv2d(hidden_dim, C, kernel_size=3, padding=1)
        self.conv_beta = nn.Conv2d(hidden_dim, C, kernel_size=3, padding=1)

        # Zero-init 전략: 학습 초기에 항등 함수로 동작하도록 0으로 초기화
        nn.init.zeros_(self.conv_gamma.weight)
        nn.init.zeros_(self.conv_gamma.bias)
        nn.init.zeros_(self.conv_beta.weight)
        nn.init.zeros_(self.conv_beta.bias)


    def forward(self, feature: torch.Tensor,
                density: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature: (B, C, H, W) — 네트워크 중간 feature
            density: (B, 1, H, W) — 구름 밀도 맵

        Returns:
            modulated: (B, C, H, W) — 변조된 feature
        """
        B, C, H, W = feature.shape

        # --------------------------------------------------------------------
        # [AS-IS] 기존 코드 (GAP 기반) - 주석 처리
        # --------------------------------------------------------------------
        # # GAP(Global Average Pooling)으로 글로벌 조건 벡터 생성
        # feat_avg = F.adaptive_avg_pool2d(feature, 1).view(B, C)  # (B, C)
        # dens_avg = F.adaptive_avg_pool2d(density, 1).view(B, 1)  # (B, 1)
        # cond = torch.cat([feat_avg, dens_avg], dim=1)             # (B, C+1)
        #
        # # γ(scale), β(shift) 생성 (모든 픽셀에 동일하게 적용됨)
        # gamma = self.scale_net(cond).view(B, C, 1, 1)  # (B, C, 1, 1)
        # beta = self.shift_net(cond).view(B, C, 1, 1)   # (B, C, 1, 1)

        # --------------------------------------------------------------------
        # [TO-BE] 신규 코드 (공간 가변 변조 - Spatial Adaptive Modulation)
        # --------------------------------------------------------------------
        # 밀도 지도를 Conv 레이어에 통과시켜 (B, C, H, W) 형태의 공간 가중치 맵 생성
        shared = self.shared_conv(density)
        
        gamma_map = self.conv_gamma(shared)  # 형태: (B, C, H, W)
        beta_map = self.conv_beta(shared)    # 형태: (B, C, H, W)

        # AdaIN 변조: 픽셀 좌표마다 1:1 매칭 연산
        # 맑은 영역은 원본 유지(gamma_map ≒ 0), 구름 영역은 강하게 변조
        modulated = feature * (1 + gamma_map) + beta_map

        return modulated


# ============================================================================
# 통합 모듈: CAFM (Cloud-Adaptive Feature Modulation)
# ============================================================================
class CAFM(nn.Module):
    """Cloud-Adaptive Feature Modulation 통합 모듈.

    ACA-CRNet의 ResBlock_att 직후에 삽입되어,
    외부에서 전달받은 밀도맵으로 feature를 적응적으로 변조한다.
    CAFM #1과 #2는 동일한 밀도맵을 공유하되 각자 다른 γ/β를 학습한다.

    Args:
        feature_channels: 변조할 feature 채널 수 (기본 256)
    """

    def __init__(self, feature_channels: int = 256):
        super().__init__()
        self.modulator = AdaptiveFeatureModulator(feature_channels)

    def forward(self, feature: torch.Tensor,
                density: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature: (B, C, H, W) — 현재 레이어의 feature
            density: (B, 1, H, W) — 구름 밀도 맵

        Returns:
            modulated: (B, C, H, W) — 변조된 feature
        """
        return self.modulator(feature, density)
