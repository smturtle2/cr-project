"""Focal Frequency Loss (단일 타겟 복원용).

CMTFusion(`CMTFusion/losses.py`)의 `frequency` 손실은 두 소스(RGB/IR)를 받는
이미지 *융합* 전용이라, 두 Gaussian 저역통과 필터로 "어느 소스가 이 주파수를
소유하는가"를 동적으로 배분한다. fdt_cca는 clear optical GT가 존재하는 *복원*
태스크이므로 배분할 두 소스가 없다 — 그래서 2-소스 경로/Gaussian/`.cuda()`를
걷어내고, 표준 Focal Frequency Loss(Jiang et al., ICCV 2021)를 단일 타겟
(예측 vs GT)으로 적응시킨다.

핵심은 **focal weight**다. 단순 spectral MSE는 진폭이 큰 저주파가 평균을
지배해 고주파(엣지·텍스처) 학습 신호가 묻힌다. focal weight는 "현재 못 맞춘
주파수일수록 큰 가중치"를 부여해(detach된 상수), 고정 필터 없이도 어려운
주파수에 학습을 집중시킨다.
"""

from __future__ import annotations

import torch
from torch import nn


class FocalFrequencyLoss(nn.Module):
    """예측과 타겟의 2D 주파수 스펙트럼 사이 focal-weighted 거리.

    Args:
        loss_weight: 최종 손실에 곱하는 스칼라 배율.
        alpha: focal weight의 지수(focusing 강도). 1.0이면 오차 크기에 선형
            비례(논문 기본값), 0이면 가중치가 모두 1이 되어 일반 spectral MSE.
        patch_factor: 입력을 patch_factor x patch_factor 패치로 나눠 패치별
            FFT를 수행. 1이면 영상 전체에 대해 한 번 FFT.
        average_spectrum: True면 미니배치 평균 스펙트럼끼리 비교.
        log_weight: True면 focal weight를 log(1+w)로 압축해 dynamic range를 완화.
        batchwise_weight: True면 focal weight 정규화를 배치 전체 최댓값 기준으로,
            False면 샘플(+패치)별 최댓값 기준으로 수행.
    """

    def __init__(
        self,
        *,
        loss_weight: float = 1.0,
        alpha: float = 1.0,
        patch_factor: int = 1,
        average_spectrum: bool = False,
        log_weight: bool = False,
        batchwise_weight: bool = False,
    ) -> None:
        super().__init__()
        if patch_factor <= 0:
            raise ValueError("patch_factor must be greater than zero")
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.average_spectrum = average_spectrum
        self.log_weight = log_weight
        self.batchwise_weight = batchwise_weight

    def _to_spectrum(self, x: torch.Tensor) -> torch.Tensor:
        # 1) patchify: 영상을 patch_factor x patch_factor 격자로 잘라 쌓는다.
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        if h % patch_factor != 0 or w % patch_factor != 0:
            raise ValueError("patch_factor must divide both image height and width")
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        patches = [
            x[:, :, i * patch_h : (i + 1) * patch_h, j * patch_w : (j + 1) * patch_w]
            for i in range(patch_factor)
            for j in range(patch_factor)
        ]
        # shape: [B, P, C, patch_h, patch_w]  (P = patch_factor**2)
        patch_stack = torch.stack(patches, dim=1)

        # 2) 2D DFT (orthonormal). 채널별로 독립 수행되므로 채널 수에 무관.
        spectrum = torch.fft.fft2(patch_stack, norm="ortho")
        # 복소수를 마지막 축의 (real, imag) 쌍으로 펼친다: [..., 2]
        return torch.stack([spectrum.real, spectrum.imag], dim=-1)

    def _weighted_distance(
        self,
        pred_spectrum: torch.Tensor,
        target_spectrum: torch.Tensor,
        precomputed_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 각 주파수에서의 제곱 유클리드 거리: (ΔRe)^2 + (ΔIm)^2
        spectrum_diff = pred_spectrum - target_spectrum
        spectrum_distance = spectrum_diff[..., 0] ** 2 + spectrum_diff[..., 1] ** 2

        if precomputed_weight is not None:
            # 미리 계산된 가중치 행렬을 쓰는 경우(고정 spectrum weighting).
            focal_weight = precomputed_weight.detach()
        else:
            # focal weight: 거리 자체로 가중치를 만든다. |diff|^alpha = distance^(alpha/2).
            # 못 맞춘 주파수(거리 큼)일수록 큰 가중치를 받는다.
            focal_weight = spectrum_distance.sqrt() ** self.alpha
            if self.log_weight:
                # 가중치 편차가 극단적일 때 range를 눌러 학습을 안정화.
                focal_weight = torch.log(focal_weight + 1.0)

            # [0, 1]로 정규화: 배치 전체 또는 샘플(+패치)별 최댓값 기준.
            if self.batchwise_weight:
                focal_weight = focal_weight / focal_weight.max()
            else:
                max_per_sample = focal_weight.amax(dim=(-2, -1), keepdim=True)
                focal_weight = focal_weight / max_per_sample

            focal_weight[torch.isnan(focal_weight)] = 0.0
            # detach: 가중치는 "어디에 집중할지"를 정하는 상수. 그래디언트가
            # 가중치를 줄여 손실을 낮추는 꼼수를 막고, 실제 거리 항으로만 흐르게 한다.
            focal_weight = torch.clamp(focal_weight, min=0.0, max=1.0).detach()

        return torch.mean(focal_weight * spectrum_distance)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        precomputed_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target shapes must match: {pred.shape} != {target.shape}"
            )
        # FFT는 fp32에서 수행해야 안정적이므로 autocast를 끄고 float로 캐스팅한다.
        with torch.autocast(device_type=pred.device.type, enabled=False):
            pred_spectrum = self._to_spectrum(pred.float())
            target_spectrum = self._to_spectrum(target.float())

            if self.average_spectrum:
                pred_spectrum = torch.mean(pred_spectrum, dim=0, keepdim=True)
                target_spectrum = torch.mean(target_spectrum, dim=0, keepdim=True)

            return self.loss_weight * self._weighted_distance(
                pred_spectrum, target_spectrum, precomputed_weight
            )
