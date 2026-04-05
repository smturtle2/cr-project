# -*- coding: utf-8 -*-
"""
SAR-Optical Cross-Modal Transformer (모듈 ①)
=============================================
[신규 파일] ACA-CRNet에 플러그인으로 장착되는 SAR↔Optical cross-modal attention.

핵심 아이디어:
    SAR 전용 light encoder로 SAR feature를 독립 추출한 뒤,
    Optical 복원 feature(Q)가 SAR 구조정보(K/V)를 채널축 attention으로 조회한다.
    residual은 Q(=F_opt)로 유지되어 optical 복원 경로에 자연스럽게 합류.

Attention 방향 (리포트 기본안):
    Q  = F_opt  (Optical — 복원 중인 feature가 능동 조회자)
    K  = F_sar  (SAR — 구조정보 제공)
    V  = F_sar
    Residual = Q (=F_opt) — optical 공간 유지

참고 논문 및 코드 출처:
    [1] AFR-CR (Remote Sensing 2026) — Transposed Cross-Attention (채널축 MDTA)
        "AFR-CR: An Adaptive Frequency Domain Feature Reconstruction-Based Method
         for Cloud Removal via SAR-Assisted Remote Sensing Image Fusion"
        Eq.(2): Q=DW(1×1(F*)), K=DW(1×1(X)), V=DW(1×1(X)), softmax(QK^T/α)V
        공간축 attention O((HW)²) 대신 채널축 O(C²)로 256×256 입력에서 OOM 회피.

    [2] W-shaped Fusion (Information Fusion 2024, DDIA-CFR)
        "Breaking through clouds: A hierarchical fusion network empowered by
         dual-domain cross-modality interactive attention"
        SAR/Optical 독립 branch encoder 구조 차용 (single-input concat의 한계 극복).

    [3] Restormer (CVPR 2022) — MDTA(Multi-Dconv head Transposed Attention) + GDFN
        https://github.com/swz30/Restormer
        1×1 pointwise + 3×3 depthwise conv projection, L2-normalized Q/K,
        learnable temperature α, gated-dwconv FFN 패턴의 원전.

    [4] EMRDM (CVPR 2025) — zero_module 초기화 전략
        플러그인의 마지막 projection을 0으로 초기화하여 학습 초기 항등함수 보장.
        (modules/model/cafm.py의 zero_module 재사용)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 유틸리티: Channel-wise LayerNorm (Restormer 스타일)
# ============================================================================
class LayerNorm2d(nn.Module):
    """(B, C, H, W) 텐서에 대해 채널축 LayerNorm을 적용한다.

    공간 위치별로 채널 간 통계를 정규화 (Restormer의 WithBias_LayerNorm과 동치).
    내부적으로 (B, H, W, C)로 permute → nn.LayerNorm(C) → 원형 복원.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, H, W, C) -> LN -> (B, C, H, W)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


# ============================================================================
# Part A: SAR 전용 Light Encoder
# 출처: W-shaped Fusion (Info. Fusion 2024) — two-branch encoder 컨셉
# ============================================================================
class SARLightEncoder(nn.Module):
    """SAR 입력(2ch)을 feature 공간(C ch)으로 매핑하는 경량 인코더.

    Optical 파이프라인과 독립적으로 SAR feature를 추출하여
    cross-attention의 K/V 소스로 제공한다.
    공간 해상도는 유지 (downsampling 없음) — ACA-CRNet body1 출력과 공간 정합 필요.

    Args:
        sar_channels: SAR 입력 채널 (기본 2: VV, VH)
        feature_channels: 출력 feature 채널 (기본 256 = ACA-CRNet feature_sizes)
        num_blocks: Conv+GELU 블록 수 (기본 2)
    """

    def __init__(self, sar_channels: int = 2, feature_channels: int = 256,
                 num_blocks: int = 2):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(sar_channels, feature_channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
        ]
        for _ in range(num_blocks - 1):
            layers += [
                nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1, bias=True),
                nn.GELU(),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, sar: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sar: (B, 2, H, W)
        Returns:
            f_sar: (B, C, H, W)
        """
        return self.net(sar)


# ============================================================================
# Part B: Transposed Cross-Modal Attention (채널축 MDTA의 cross-modal 확장)
# 출처: AFR-CR (Remote Sensing 2026) CFRM Eq.(2) + Restormer MDTA
# ============================================================================
class CrossModalMDTA(nn.Module):
    """Multi-Dconv head Transposed Cross-Modal Attention.

    Restormer MDTA를 cross-modal로 확장: Q와 K/V를 서로 다른 modality에서 추출한다.
    채널축 attention이므로 복잡도는 O(C²)로 256×256 입력에서도 메모리 안전.

    수식 (리포트 모듈 ① 기본안):
        Q = DWConv(1×1(F_opt))           ─ Optical이 능동 조회자
        K = DWConv(1×1(F_sar))           ─ SAR이 구조 key
        V = DWConv(1×1(F_sar))           ─ SAR이 값
        attn = softmax(Q_norm · K_norm^T · α)
        out  = attn · V  →  project_out(out)

    Residual(상위 CrossModalBlock에서 처리)은 Q(=F_opt)에 적용되어 optical 공간 유지.

    Args:
        dim: feature 채널 수 (Q/K/V 공통)
        num_heads: multi-head 수 (dim이 num_heads로 나누어 떨어져야 함)
        bias: 1×1 / depthwise conv의 bias 사용 여부
    """

    def __init__(self, dim: int, num_heads: int = 4, bias: bool = False):
        super().__init__()
        assert dim % num_heads == 0, f"dim({dim})은 num_heads({num_heads})로 나누어 떨어져야 한다"
        self.num_heads = num_heads
        # head별 학습 가능한 temperature (Restormer 관례)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # Q: F_opt에서만 생성
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias)

        # K, V: F_sar에서 공동 생성 (효율 위해 2*dim로 한 번에)
        self.kv_proj = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, groups=dim * 2, bias=bias)

        # 출력 projection (zero-init 대상)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, f_opt: torch.Tensor, f_sar: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_opt: (B, C, H, W) — Query 소스
            f_sar: (B, C, H, W) — Key/Value 소스
        Returns:
            out: (B, C, H, W) — attention 결과 (residual은 호출자 쪽에서 더함)
        """
        b, c, h, w = f_opt.shape
        head_dim = c // self.num_heads

        # 프로젝션: 1×1 pointwise → 3×3 depthwise (채널간 상호작용 + 지역성)
        q = self.q_dwconv(self.q_proj(f_opt))             # (B, C, H, W)
        kv = self.kv_dwconv(self.kv_proj(f_sar))           # (B, 2C, H, W)
        k, v = kv.chunk(2, dim=1)                          # 각 (B, C, H, W)

        # head 분할 + 공간 flatten: (B, heads, head_dim, HW)
        q = q.view(b, self.num_heads, head_dim, h * w)
        k = k.view(b, self.num_heads, head_dim, h * w)
        v = v.view(b, self.num_heads, head_dim, h * w)

        # Restormer 관례: Q, K를 공간축으로 L2 정규화 (temperature와 결합해 스케일 안정)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # 채널축 attention: (B, heads, head_dim, head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # attention 적용: (B, heads, head_dim, HW)
        out = attn @ v
        out = out.contiguous().view(b, c, h, w)
        return self.project_out(out)


# ============================================================================
# Part C: Gated-Dconv FFN (Restormer GDFN)
# ============================================================================
class GDFN(nn.Module):
    """Gated Depthwise-conv Feed-Forward Network (Restormer).

    표준 MLP의 두 분기를 gating으로 결합하여 feature selection 능력 강화.
    Conv2d 기반이라 2D feature map에 그대로 적용 가능.

    Args:
        dim: 입력/출력 채널
        expansion: hidden 확장 배수 (기본 2.0)
        bias: conv bias 사용 여부
    """

    def __init__(self, dim: int, expansion: float = 2.0, bias: bool = False):
        super().__init__()
        hidden = int(dim * expansion)
        self.proj_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1,
                                groups=hidden * 2, bias=bias)
        self.proj_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dwconv(self.proj_in(x))
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.proj_out(x)


# ============================================================================
# Part D: Cross-Modal Transformer Block (통합 모듈 — 외부 플러그인 API)
# ============================================================================
class CrossModalBlock(nn.Module):
    """SAR→Optical Cross-Modal Transformer Block.

    pre-LN 구조 (Restormer와 동일):
        f_opt ← f_opt + CrossModalMDTA(LN(f_opt), LN(f_sar))
        f_opt ← f_opt + GDFN(LN(f_opt))

    zero_module을 통해 project_out / proj_out이 0-init되므로 학습 초기에
    attention/FFN 출력이 0이 되어 블록 전체가 항등함수로 시작한다.
    이는 팀 CAFM 모듈과 동일한 전략으로, 베이스라인 성능을 초기에 해치지 않는다.

    Args:
        dim: feature 채널 수
        num_heads: attention head 수
        ffn_expansion: GDFN hidden 확장 배수
        bias: conv bias 사용 여부
    """

    def __init__(self, dim: int, num_heads: int = 4,
                 ffn_expansion: float = 2.0, bias: bool = False):
        super().__init__()
        self.norm_opt = LayerNorm2d(dim)
        self.norm_sar = LayerNorm2d(dim)
        self.attn = CrossModalMDTA(dim, num_heads=num_heads, bias=bias)
        self.norm_ffn = LayerNorm2d(dim)
        self.ffn = GDFN(dim, expansion=ffn_expansion, bias=bias)

    def forward(self, f_opt: torch.Tensor, f_sar: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_opt: (B, C, H, W) — ACA-CRNet body1 출력 (복원 중인 optical feature)
            f_sar: (B, C, H, W) — SARLightEncoder 출력
        Returns:
            f_opt: (B, C, H, W) — SAR 정보가 주입된 optical feature
        """
        # Cross-modal attention: residual이 f_opt(Q)에 적용 → optical 공간 유지
        f_opt = f_opt + self.attn(self.norm_opt(f_opt), self.norm_sar(f_sar))
        # FFN: 각 modality feature가 아닌 변조된 optical feature만 정제
        f_opt = f_opt + self.ffn(self.norm_ffn(f_opt))
        return f_opt
