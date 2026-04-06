"""Cloud removal 평가 지표 (SEN12MS-CR 표준).

- MAE: Mean Absolute Error (값이 작을수록 좋음)
- PSNR: Peak Signal-to-Noise Ratio (dB, 값이 클수록 좋음)
- SSIM: Structural Similarity Index (값이 클수록 좋음)
- SAM: Spectral Angle Mapper (degree, 값이 작을수록 좋음)

데이터 스케일 주의:
    cr-train이 Sentinel-2 optical을 clamp(0, 10000) 후 1/2000로 정규화한다.
    (cr_train/data/dataset.py 참조)
    즉 모델 입출력의 실제 값 범위는 대략 [0, 5]이고,
    PSNR/SSIM의 `data_range`도 이 스케일에 맞춰야 한다.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import torch
import torch.nn.functional as F


Batch = dict[str, Any]
MetricFn = Callable[[torch.Tensor, Batch], torch.Tensor]

# cr-train이 Sentinel-2 반사도 [0, 10000]를 *1/2000 정규화하므로
# 메트릭이 보는 유효 값 범위는 [0, 5]다.
DEFAULT_DATA_RANGE: float = 5.0


def mae(prediction: torch.Tensor, batch: Batch) -> torch.Tensor:
    """Mean Absolute Error. 픽셀 단위 평균 절대 오차."""
    return torch.mean(torch.abs(prediction - batch["target"]))


def psnr(
    prediction: torch.Tensor,
    batch: Batch,
    data_range: float = DEFAULT_DATA_RANGE,
) -> torch.Tensor:
    """Peak Signal-to-Noise Ratio.

    cr-train의 optical 정규화([0, 10000] → [0, 5])에 맞춰 `data_range=5.0` 사용.
    값이 클수록 좋음.
    """
    mse = torch.mean((prediction - batch["target"]) ** 2)
    mse = torch.clamp(mse, min=1e-10)
    return 10.0 * torch.log10((data_range ** 2) / mse)


def _gaussian_window(window_size: int, sigma: float) -> torch.Tensor:
    """SSIM 계산용 2D Gaussian 커널 생성."""
    coords = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    window_2d = g.unsqueeze(1) @ g.unsqueeze(0)  # (window_size, window_size)
    return window_2d


def ssim(
    prediction: torch.Tensor,
    batch: Batch,
    data_range: float = DEFAULT_DATA_RANGE,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Structural Similarity Index.

    Gaussian window 기반 SSIM을 각 채널/위치에 대해 계산 후 평균.
    값이 클수록 좋음 (최대 1.0).

    `data_range`는 cr-train 정규화 범위와 일치해야 C1/C2 상수가 의미를 가진다.
    pred, target: (B, C, H, W)
    """
    target = batch["target"]
    C = prediction.shape[1]
    device = prediction.device
    dtype = prediction.dtype

    # Gaussian window를 채널마다 준비 (depthwise conv 형태)
    window = _gaussian_window(window_size, sigma).to(device=device, dtype=dtype)
    window = window.expand(C, 1, window_size, window_size).contiguous()

    pad = window_size // 2

    # 국소 평균
    mu1 = F.conv2d(prediction, window, padding=pad, groups=C)
    mu2 = F.conv2d(target, window, padding=pad, groups=C)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    # 국소 분산/공분산
    sigma1_sq = F.conv2d(prediction * prediction, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(prediction * target, window, padding=pad, groups=C) - mu1_mu2

    # SSIM 상수 (데이터 범위에 비례)
    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    ssim_map = ((2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


def sam(prediction: torch.Tensor, batch: Batch) -> torch.Tensor:
    """Spectral Angle Mapper.

    각 픽셀의 스펙트럼 벡터(채널 방향) 사이의 각도(degree)를 계산.
    원격탐사 multi-band 영상에서 스펙트럼 정확도를 측정.
    값이 작을수록 좋음. (SEN12MS-CR/ACA-CRNet 문헌 컨벤션: degree)

    pred, target: (B, C, H, W) - C는 채널(밴드) 수
    """
    target = batch["target"]

    # 채널(C) 축으로 내적과 norm 계산
    dot = torch.sum(prediction * target, dim=1)            # (B, H, W)
    norm_pred = torch.sqrt(torch.sum(prediction ** 2, dim=1))  # (B, H, W)
    norm_target = torch.sqrt(torch.sum(target ** 2, dim=1))    # (B, H, W)

    # 분모 안정화 (0 벡터 방지)
    denom = torch.clamp(norm_pred * norm_target, min=1e-8)

    # cos 값 계산 후 acos 범위 안정화
    cos_sim = dot / denom
    cos_sim = torch.clamp(cos_sim, min=-1.0 + 1e-7, max=1.0 - 1e-7)

    angle = torch.acos(cos_sim)  # (B, H, W), 라디안

    # 라디안 → 도 변환 (문헌 컨벤션)
    return torch.mean(angle) * (180.0 / math.pi)


def build_metrics() -> dict[str, MetricFn]:
    """Trainer에 넘길 metric 딕셔너리.

    SEN12MS-CR 표준 4종: SSIM, MAE, SAM, PSNR
    """
    return {
        "mae": mae,
        "psnr": psnr,
        "ssim": ssim,
        "sam": sam,
    }


__all__ = [
    "build_metrics",
    "mae",
    "psnr",
    "ssim",
    "sam",
    "MetricFn",
    "DEFAULT_DATA_RANGE",
]
