"""AFR-CR (Adaptive Frequency domain feature Reconstruction for Cloud Removal)의
주파수 도메인 핵심 모듈.

핵심 아이디어:
  1. 2D FFT로 feature를 주파수 도메인으로 변환 -> 진폭(amplitude) + 위상(phase)
  2. 학습 가능한 동적 마스크로 저주파/고주파를 적응적으로 분리
     -> 구름은 저주파에, 지표면 디테일은 고주파에 주로 존재
     -> 구름이 두꺼울수록 저주파 에너지가 커지므로 마스크가 더 넓게 저주파를 잡게 된다.
  3. SAR 주파수 특징과 optical 주파수 특징을 교차 재구성
  4. IFFT로 다시 공간 도메인으로 복원
"""

from __future__ import annotations

import torch
from torch import nn


class FrequencyDecomposition(nn.Module):
    """2D FFT로 feature map을 진폭/위상으로 분해하고,
    학습 가능한 동적 마스크로 저주파/고주파를 적응적으로 분리한다.

    AFR-CR의 FDDM(Adaptive Frequency Domain Decoupling Module) 핵심 구현.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        # 동적 마스크 생성기: 진폭 분포를 보고 저주파/고주파 경계를 학습적으로 결정.
        # 구름이 두꺼우면 저주파 에너지가 커지므로 마스크가 더 넓게 저주파를 잡게 된다.
        self.mask_generator = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels // 2, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, C, H, W) 실수 feature map

        # Step 1: 2D FFT -> 복소수 텐서 (B, C, H, W)
        freq = torch.fft.fft2(x, norm="ortho")
        freq_shifted = torch.fft.fftshift(freq, dim=(-2, -1))  # 저주파를 중심으로 이동

        # Step 2: 진폭(amplitude)과 위상(phase) 분리
        amplitude = torch.abs(freq_shifted)     # "그 주파수 성분이 얼마나 강한가"
        phase = torch.angle(freq_shifted)        # "그 주파수 패턴이 어디에 위치하는가"

        # Step 3: 동적 마스크 생성 (진폭 기반)
        # 각 채널/위치마다 "이게 저주파(구름)인지 고주파(지표면)인지" 결정
        mask_low = self.mask_generator(amplitude)  # (B, C, H, W), 0~1
        mask_high = 1.0 - mask_low

        # Step 4: 진폭을 저주파/고주파로 분리
        amp_low = amplitude * mask_low    # 저주파 성분 (구름에 해당)
        amp_high = amplitude * mask_high  # 고주파 성분 (지표면에 해당)

        return amp_low, amp_high, phase, mask_low


class FrequencyReconstruction(nn.Module):
    """분리된 주파수 성분을 IFFT로 공간 도메인 feature로 되돌린다.

    AFR-CR의 FCFR(Frequency-domain Cross-Feature Reconstruction) 합성 단계.
    """

    @staticmethod
    def forward(amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        # 복소수 재구성: real = amplitude * cos(phase), imag = amplitude * sin(phase)
        real = amplitude * torch.cos(phase)
        imag = amplitude * torch.sin(phase)
        freq_shifted = torch.complex(real, imag)

        # 역 FFT로 공간 도메인 복원
        freq = torch.fft.ifftshift(freq_shifted, dim=(-2, -1))
        out = torch.fft.ifft2(freq, norm="ortho")
        return out.real  # 실수부만 취함


class CrossFrequencyFusion(nn.Module):
    """SAR 주파수 특징과 optical 주파수 특징을 교차 재구성하여 융합한다.

    AFR-CR의 핵심 아이디어:
      - SAR은 구름 투과 -> 구름 없는 저주파 정보 제공
      - Optical은 색상/텍스처 제공 (고주파)
      - 저주파는 SAR 기반으로 보강, 고주파는 optical 기반으로 유지
      - Optical의 위상(위치 정보)을 보존하여 물체 위치가 어긋나지 않게 함
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.decompose_opt = FrequencyDecomposition(channels)
        self.decompose_sar = FrequencyDecomposition(channels)
        self.reconstruct = FrequencyReconstruction()

        # 저주파 교차 재구성: SAR 저주파로 optical 저주파를 보정
        # 구름에 오염된 optical 저주파를 SAR 저주파 참조로 정제
        self.low_fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

        # 고주파 정제: optical 고주파 유지 + SAR 고주파로 구조 보강
        self.high_fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, opt_feat: torch.Tensor, sar_feat: torch.Tensor) -> torch.Tensor:
        # optical의 주파수 분해
        opt_amp_low, opt_amp_high, opt_phase, _ = self.decompose_opt(opt_feat)
        # SAR의 주파수 분해
        sar_amp_low, sar_amp_high, _, _ = self.decompose_sar(sar_feat)

        # 저주파 교차 재구성: SAR의 구름 없는 저주파 정보로 optical 보정
        fused_amp_low = self.low_fusion(torch.cat([opt_amp_low, sar_amp_low], dim=1))

        # 고주파: optical 우선 유지, SAR로 구조 보강
        fused_amp_high = self.high_fusion(torch.cat([opt_amp_high, sar_amp_high], dim=1))

        # 합친 진폭 + optical의 원래 위상(위치 정보)을 사용해 복원
        fused_amp = fused_amp_low + fused_amp_high
        out = self.reconstruct(fused_amp, opt_phase)
        return out
