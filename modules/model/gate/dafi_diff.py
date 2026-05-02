from __future__ import annotations

import torch
from torch import nn

from .base import BaseGateEstimator


class _FeatureEncoder(nn.Module):
    def __init__(self, in_channels: int, feat_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, feat_dim, kernel_size=1, bias=True),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ContextGate(nn.Module):
    def __init__(self, feat_dim: int) -> None:
        super().__init__()
        self.context = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim, bias=True),
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1, bias=True),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(feat_dim * 2, feat_dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, value: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        # guide가 만든 문맥 가중치로 value를 재가중한다.
        context = self.context(guide)
        weight = self.gate(torch.cat([value, context], dim=1))
        return value * weight


class _DifferentialAttentionProxy(nn.Module):
    def __init__(self, feat_dim: int) -> None:
        super().__init__()
        self.optical_self = _ContextGate(feat_dim)
        self.sar_guided = _ContextGate(feat_dim)

    def forward(self, sar_feat: torch.Tensor, optical_feat: torch.Tensor) -> torch.Tensor:
        # DAFI의 Opt_o - Opt_s를 gate 전용 proxy로 계산한다.
        opt_self = self.optical_self(optical_feat, optical_feat)
        sar_guided = self.sar_guided(optical_feat, sar_feat)
        return (opt_self - sar_guided).abs()


class _GateRefiner(nn.Module):
    def __init__(self, feat_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(feat_dim * 3 + 1, feat_dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim, bias=True),
            nn.Conv2d(feat_dim, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(
        self,
        sar_feat: torch.Tensor,
        optical_feat: torch.Tensor,
        diff_feat: torch.Tensor,
    ) -> torch.Tensor:
        # diff_map은 위치별 복원 필요도를 한 채널로 압축한 신호다.
        diff_map = diff_feat.mean(dim=1, keepdim=True)
        features = torch.cat([sar_feat, optical_feat, diff_feat, diff_map], dim=1)
        return self.net(features)


class DafiDiffGateEstimator(BaseGateEstimator):
    """DAFI differential signal을 이용한 SAR 주입 강도 gate."""

    def __init__(
        self,
        sar_channels: int = 2,
        optical_channels: int = 13,
        feat_dim: int = 32,
    ) -> None:
        super().__init__()
        self.sar_encoder = _FeatureEncoder(sar_channels, feat_dim)
        self.optical_encoder = _FeatureEncoder(optical_channels, feat_dim)
        self.diff_proxy = _DifferentialAttentionProxy(feat_dim)
        self.refine = _GateRefiner(feat_dim)

    def encode(
        self,
        sar: torch.Tensor,
        optical: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sar_feat = self.sar_encoder(sar)
        optical_feat = self.optical_encoder(optical)
        diff_feat = self.diff_proxy(sar_feat, optical_feat)
        return sar_feat, optical_feat, diff_feat

    def forward(self, sar: torch.Tensor, optical: torch.Tensor) -> torch.Tensor:
        sar_feat, optical_feat, diff_feat = self.encode(sar, optical)
        return self.refine(sar_feat, optical_feat, diff_feat)
