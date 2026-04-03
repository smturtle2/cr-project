# -*- coding: utf-8 -*-
"""
ACA-CRNet + CAFM (Cloud-Adaptive Feature Modulation)
=====================================================
원본 ACA-CRNet에 CAFM 모듈을 플러그인으로 장착한 버전.
cr-train의 forward(sar, cloudy) 인터페이스에 맞게 수정.

[수정 사항] ← 이 표시가 있는 부분이 원본 대비 변경/추가된 코드
  1) forward(sar, cloudy) 시그니처 — cr-train 호환
  2) nn.Sequential → body1/body2/body3 분리 — CAFM 삽입 지점 확보
  3) Optical-only 글로벌 잔차 — SAR은 잔차에서 제외
  4) CAFM 모듈 장착 — ResBlock_att 직후 2곳
  5) self.last_density / self.last_cloudy — 커스텀 Loss에서 참조

원본 코드 출처:
    @author: Wenli Huang
    @paper: "Attentive Contextual Attention for Cloud Removal", IEEE TGRS, 2024
    @code: https://github.com/huangwenwenlili/ACA-CRNet
    기반: DSEN2_CR_PYTORCH (https://github.com/Phoenix-Shen/DSEN2_CR_PYTORCH)
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from torch.nn import init

from .ca import ConAttn
from .cafm import CloudDensityEstimator, CAFM  # [수정] CAFM 모듈 임포트


# ============================================================================
# 원본 유지: ResBlock, ResBlock_att, init_weights
# ============================================================================
class ResBlock(nn.Module):
    """원본 ACA-CRNet ResBlock — 수정 없음."""
    def __init__(self, in_channels, out_channels=256, alpha=0.1):
        super(ResBlock, self).__init__()
        m = OrderedDict()
        m['conv1'] = nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False, stride=1, padding=1)
        m['relu1'] = nn.ReLU(True)
        m['conv2'] = nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False, stride=1, padding=1)
        self.net = nn.Sequential(m)
        self.relu = nn.Sequential(nn.ReLU(True))
        self.alpha = alpha

    def forward(self, x):
        out = self.net(x)
        out = self.alpha * out + x
        return out


class ResBlock_att(nn.Module):
    """원본 ACA-CRNet ResBlock with AC-Attention — 수정 없음."""
    def __init__(self, in_channels, out_channels=256, alpha=0.1):
        super(ResBlock_att, self).__init__()
        m = OrderedDict()
        m['conv1'] = nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False, stride=2, padding=1)
        m['relu1'] = nn.ReLU(True)
        m['conv2'] = nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False, stride=1, padding=1)
        m['relu2'] = nn.ReLU(True)
        m['att'] = ConAttn(input_channels=out_channels, output_channels=out_channels, ksize=1, stride=1)
        self.net = nn.Sequential(m)
        self.alpha = alpha

    def forward(self, x):
        out = self.net(x)
        out = torch.nn.functional.interpolate(out, scale_factor=2, mode='bilinear',
                                              align_corners=True)
        out = self.alpha * out + x
        return out


def init_weights(net, init_type="kaiming-uniform", gain=0.02):
    """원본 가중치 초기화 함수 — 수정 없음."""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == "kaiming-uniform":
                init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


# ============================================================================
# [수정] ACA_CRNet: cr-train 호환 + CAFM 장착
# ============================================================================
class ACA_CRNet(nn.Module):
    """ACA-CRNet + CAFM for SEN12MS-CR (cr-train 호환).

    변경점 (원본 대비):
        1) forward(sar, cloudy) — cr-train이 model(*inputs)로 호출
        2) in_channels=15 (S2 13ch + S1 2ch), out_channels=13 (S2 복원)
        3) body1/body2/body3 분리 — CAFM 삽입 지점
        4) Optical-only 잔차 — cloudy + net_output
        5) use_cafm 플래그 — False면 원본과 동일 동작 (Ablation용)

    Args:
        sar_channels: S1 SAR 채널 수 (기본 2)
        opt_channels: S2 Optical 채널 수 (기본 13)
        alpha: ResBlock 잔차 스케일링 (원본 기본값 0.1)
        num_layers: ResBlock 총 개수 (원본 기본값 16)
        feature_sizes: 내부 feature 채널 수 (원본 기본값 256)
        use_cafm: CAFM 모듈 사용 여부 (False면 원본 동작)
        cafm_feat_dim: CAFM 밀도 추정 프로젝션 차원 (기본 32)
    """

    def __init__(self, sar_channels: int = 2, opt_channels: int = 13,
                 alpha: float = 0.1, num_layers: int = 16,
                 feature_sizes: int = 256,
                 use_cafm: bool = True,       # [수정] CAFM on/off
                 cafm_feat_dim: int = 32):    # [수정] 밀도 추정 차원
        super(ACA_CRNet, self).__init__()

        self.use_cafm = use_cafm
        self.opt_channels = opt_channels
        in_channels = sar_channels + opt_channels  # [수정] 15ch 입력

        # ── Head: 15ch → 256ch ──
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, feature_sizes, kernel_size=3, bias=True, stride=1, padding=1),
            nn.ReLU(True),
        )

        # ── Body: 3구간 분리 (원본은 단일 Sequential) ──
        # [수정] CAFM 삽입을 위해 ResBlock_att 경계에서 분리
        # 구간 1: layer 0 ~ num_layers//2 (마지막이 ResBlock_att)
        body1 = []
        for i in range(num_layers // 2 + 1):
            if i == num_layers // 2:
                body1.append(ResBlock_att(feature_sizes, feature_sizes, alpha))
            else:
                body1.append(ResBlock(feature_sizes, feature_sizes, alpha))
        self.body1 = nn.Sequential(*body1)

        # 구간 2: layer num_layers//2+1 ~ num_layers*3//4 (마지막이 ResBlock_att)
        body2 = []
        for i in range(num_layers // 2 + 1, num_layers * 3 // 4 + 1):
            if i == num_layers * 3 // 4:
                body2.append(ResBlock_att(feature_sizes, feature_sizes, alpha))
            else:
                body2.append(ResBlock(feature_sizes, feature_sizes, alpha))
        self.body2 = nn.Sequential(*body2)

        # 구간 3: 나머지 ResBlock들
        body3 = []
        for i in range(num_layers * 3 // 4 + 1, num_layers):
            body3.append(ResBlock(feature_sizes, feature_sizes, alpha))
        self.body3 = nn.Sequential(*body3)

        # ── Tail: 256ch → 13ch (S2 전 밴드 복원) ──
        self.tail = nn.Conv2d(feature_sizes, opt_channels, kernel_size=3, bias=True, stride=1, padding=1)

        # ── [수정] CAFM 모듈 ──
        if use_cafm:
            # Part A: SAR-Optical 코사인 유사도 기반 밀도 추정
            self.density_estimator = CloudDensityEstimator(
                sar_channels=sar_channels,
                optical_channels=opt_channels,
                feat_dim=cafm_feat_dim,
            )
            # Part B: AdaIN-style feature 변조 (2개 — layer 8, 12 직후)
            self.cafm1 = CAFM(feature_sizes)  # body1(ResBlock_att #1) 직후
            self.cafm2 = CAFM(feature_sizes)  # body2(ResBlock_att #2) 직후

        # [수정] 커스텀 Loss/시각화에서 참조할 속성
        self.last_density = None   # 마지막 forward의 밀도맵
        self.last_cloudy = None    # 마지막 forward의 cloudy 입력

        # 가중치 초기화 (원본 방식 유지)
        init_weights(self, "kaiming-uniform")

        # [수정] zero_module 재적용: init_weights가 CAFM의 zero-init을 덮어쓰므로
        # 마지막 Linear 레이어를 다시 0으로 초기화하여 학습 초기 항등 함수 보장
        if use_cafm:
            from .cafm import zero_module
            for cafm_module in [self.cafm1, self.cafm2]:
                zero_module(cafm_module.modulator.scale_net[-1])
                zero_module(cafm_module.modulator.shift_net[-1])

    def forward(self, sar: torch.Tensor, cloudy: torch.Tensor) -> torch.Tensor:
        """cr-train 호환 forward.

        cr-train Trainer가 model(*batch["inputs"])로 호출한다.
        batch["inputs"] = (sar, cloudy) 튜플이 언팩되어 전달됨.

        Args:
            sar:    (B, 2, 256, 256)  — Sentinel-1 VV/VH, [0, 1]
            cloudy: (B, 13, 256, 256) — Sentinel-2 Cloudy, [0, 1]

        Returns:
            pred: (B, 13, 256, 256) — Cloud-Free 예측, [0, 1] 범위
        """
        # [수정] cloudy 저장 — CloudAdaptiveLoss에서 참조
        self.last_cloudy = cloudy

        # ── [수정] Part A: 밀도 추정 (Head Conv 전, 원본 입력에서) ──
        if self.use_cafm:
            self.last_density = self.density_estimator(sar, cloudy)
        else:
            self.last_density = None

        # ── 입력 결합: concat(cloudy, sar) = 15ch ──
        x = torch.cat([cloudy, sar], dim=1)  # (B, 15, H, W)

        # ── Head ──
        feat = self.head(x)

        # ── Body 구간 1 → ResBlock_att (layer 8) ──
        feat = self.body1(feat)
        # [수정] CAFM #1: 중간 feature 변조
        if self.use_cafm:
            feat = self.cafm1(feat, self.last_density)

        # ── Body 구간 2 → ResBlock_att (layer 12) ──
        feat = self.body2(feat)
        # [수정] CAFM #2: 후반 feature 변조 (같은 밀도맵, 다른 γ/β)
        if self.use_cafm:
            feat = self.cafm2(feat, self.last_density)

        # ── Body 구간 3 ──
        feat = self.body3(feat)

        # ── Tail ──
        out = self.tail(feat)

        # ── [수정] Optical-only 글로벌 잔차 ──
        # 원본: return x + self.net(x) (전체 입력 잔차)
        # 수정: cloudy(13ch)만 잔차 연결. SAR ≠ Optical이므로 SAR은 잔차에 포함 불가
        pred = cloudy + out

        return pred
