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
import torch.nn.functional as F


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
# Part A: 구름 밀도 추정 (Cloud Density Estimation)
# ============================================================================
class CloudDensityEstimator(nn.Module):
    """SAR-Optical 코사인 유사도 기반 구름 밀도 추정기.

    물리적 원리 (SMDCNet, ISPRS 2025에서 검증):
        - 맑은 영역: SAR이 본 지표면 ≈ Optical이 본 지표면 → 유사도 높음 → 밀도 낮음
        - 구름 영역: SAR은 지표면, Optical은 구름 → 유사도 낮음 → 밀도 높음

    구현:
        1) SAR(2ch)과 Optical(13ch)을 동일 차원(C_d)으로 프로젝션
        2) 채널 축 코사인 유사도 계산
        3) 유사도 + feature를 결합하여 경량 네트워크로 밀도맵 정제

    Args:
        sar_channels: SAR 입력 채널 수 (기본 2: VV, VH)
        optical_channels: Optical 입력 채널 수 (기본 13: S2 전 밴드)
        feat_dim: 프로젝션 차원 C_d (기본 32)

    입력: sar (B, 2, H, W), optical (B, 13, H, W)
    출력: density (B, 1, H, W), 범위 [0, 1]
    """

    def __init__(self, sar_channels: int = 2, optical_channels: int = 13,
                 feat_dim: int = 32):
        super().__init__()
        C_d = feat_dim

        # SAR/Optical을 동일 차원으로 프로젝션 (1x1 Conv)
        self.sar_encoder = nn.Sequential(
            nn.Conv2d(sar_channels, C_d, kernel_size=1, bias=True),
            nn.GELU(),
        )
        self.opt_encoder = nn.Sequential(
            nn.Conv2d(optical_channels, C_d, kernel_size=1, bias=True),
            nn.GELU(),
        )

        # 밀도 정제 네트워크: 유사도(1ch) + SAR feature(C_d) + Opt feature(C_d) → 밀도(1ch)
        # 코사인 유사도만으로는 SAR 스펙클 노이즈에 취약하므로 feature와 결합하여 정제
        self.refine = nn.Sequential(
            nn.Conv2d(2 * C_d + 1, C_d, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(C_d, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),  # 출력 범위 [0, 1]
        )

    def forward(self, sar: torch.Tensor, optical: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sar:     (B, 2, H, W)  — Sentinel-1 VV/VH, 정규화된 [0, 1]
            optical: (B, 13, H, W) — Sentinel-2 Cloudy, 정규화된 [0, 1]

        Returns:
            density: (B, 1, H, W) — 구름 밀도 맵, 0.0(맑음) ~ 1.0(두꺼운 구름)
        """
        f_sar = self.sar_encoder(sar)       # (B, C_d, H, W)
        f_opt = self.opt_encoder(optical)   # (B, C_d, H, W)

        # 채널 축 코사인 유사도 (SMDCNet 원리)
        # 맑은 영역: ~+0.8, 구름 영역: ~-0.2
        sim = F.cosine_similarity(f_sar, f_opt, dim=1, eps=1e-6)  # (B, H, W)
        sim = sim.unsqueeze(1)  # (B, 1, H, W)

        # 유사도 + 양쪽 feature를 결합하여 밀도맵 정제
        combined = torch.cat([f_sar, f_opt, sim], dim=1)  # (B, 2*C_d+1, H, W)
        density = self.refine(combined)  # (B, 1, H, W)

        return density


# ============================================================================
# Part B: 적응적 Feature 변조 (Adaptive Feature Modulation)
# ============================================================================
class AdaptiveFeatureModulator(nn.Module):
    """AdaIN-style 채널별 Feature 변조기.

    수식: output = feature * (1 + γ) + β
        - γ(scale), β(shift)는 밀도 조건에서 생성
        - γ=0, β=0일 때 항등 함수 (맑은 영역 보존)
        - γ>0, β>0일 때 feature 강하게 변형 (구름 영역 재구성)

    설계 근거:
        [1] MWFormer (TIP 2024) FilmBlock: y_weight * x + y_bias 패턴
            코드: https://github.com/taco-group/MWFormer/blob/main/model/EncDec.py
        [2] EMRDM (CVPR 2025) ResBlock: (1 + scale) * h + shift 패턴
            코드: https://github.com/Ly403/EMRDM/blob/main/sgm/modules/diffusionmodules/openaimodel.py
        [3] (1+γ) 형태는 EMRDM의 패턴을 따름 — γ=0이면 항등함수가 되어 잔차 학습에 유리

    Zero-init 전략:
        γ/β 생성 네트워크의 마지막 레이어를 0으로 초기화 (zero_module).
        학습 초기에 CAFM이 항등 함수로 동작하여 기존 네트워크 성능을 해치지 않는다.
        출처: EMRDM의 zero_module 패턴

    Args:
        feature_channels: 변조할 feature의 채널 수 (기본 256 = ACA-CRNet feature_sizes)
    """

    def __init__(self, feature_channels: int = 256):
        super().__init__()
        C = feature_channels

        # 조건 인코딩: GAP된 feature(C) + GAP된 density(1) = C+1 차원
        # MWFormer의 in_project_y → w_project_y/b_project_y 패턴을 경량화
        self.scale_net = nn.Sequential(
            nn.Linear(C + 1, C // 4),
            nn.GELU(),
            zero_module(nn.Linear(C // 4, C)),  # zero-init: 초기 γ=0
        )
        self.shift_net = nn.Sequential(
            nn.Linear(C + 1, C // 4),
            nn.GELU(),
            zero_module(nn.Linear(C // 4, C)),  # zero-init: 초기 β=0
        )

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

        # GAP(Global Average Pooling)으로 글로벌 조건 벡터 생성
        feat_avg = F.adaptive_avg_pool2d(feature, 1).view(B, C)  # (B, C)
        dens_avg = F.adaptive_avg_pool2d(density, 1).view(B, 1)  # (B, 1)
        cond = torch.cat([feat_avg, dens_avg], dim=1)             # (B, C+1)

        # γ(scale), β(shift) 생성
        gamma = self.scale_net(cond).view(B, C, 1, 1)  # (B, C, 1, 1)
        beta = self.shift_net(cond).view(B, C, 1, 1)   # (B, C, 1, 1)

        # AdaIN 변조: EMRDM 패턴 (1 + scale) * h + shift
        modulated = feature * (1 + gamma) + beta

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
