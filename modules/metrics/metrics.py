import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# cr-train 메트릭 인터페이스:
#   - nn.Module 상속, name 속성 정의
#   - forward(outputs, target) -> scalar 텐서
#   - Trainer(..., metrics=[MAE(), PSNR(), ...]) 형태로 전달


class MAE(nn.Module):
    """Mean Absolute Error — 예측과 정답 사이 절대 오차의 평균."""

    # 출처: cr-train 내장 MAE (F.l1_loss)
    #
    # 개념: 각 픽셀에서 |예측 - 정답|을 구한 뒤 전체 평균.
    #       값이 작을수록 예측이 정답에 가깝다는 뜻.
    #       가장 직관적인 오차 지표.
    #
    # 수식: MAE = (1/N) * Σ|pred - target|

    name = "mae"

    def forward(self, outputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(outputs - target))


class PSNR(nn.Module):
    """Peak Signal-to-Noise Ratio (dB) — 높을수록 좋음."""

    # 출처: ACA-CRNet/utils/np_metric.py → cloud_psnr()
    #
    # 개념: 이미지가 가질 수 있는 최대 신호 대비 오차(잡음)의 비율을
    #       데시벨(dB) 단위로 표현한 것.
    #       MSE가 작을수록 PSNR이 높아지며, 높을수록 복원 품질이 좋음.
    #
    # 수식: PSNR = 20 * log10(MAX_VAL / sqrt(MSE))
    #       - MAX_VAL = 10000 (Sentinel-2 반사도 최대값)
    #       - MSE = mean((pred - target)^2)
    #
    # SEN12MS-CR 데이터는 로딩 시 /2000 정규화됨.
    # 계산 전 *2000으로 원래 스케일(0~10000) 복원 필요.

    name = "psnr"

    def forward(self, outputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 스케일 복원: 정규화된 값 → 원래 반사도 [0,~10000]
        pred = outputs * 2000.0
        gt = target * 2000.0

        mse = torch.mean((pred - gt) ** 2)
        if mse == 0:
            return torch.tensor(100.0, device=outputs.device)

        return 20.0 * torch.log10(
            torch.tensor(10000.0, device=outputs.device) / torch.sqrt(mse)
        )


class SSIM(nn.Module):
    """Structural Similarity Index — 1에 가까울수록 좋음."""

    # 출처: ACA-CRNet/utils/metrics_glf_cr.py → SSIM()
    #
    # 개념: 두 이미지의 구조적 유사도를 측정하는 지표.
    #       단순 픽셀 차이(MAE/PSNR)와 달리 사람의 시각 인지에 가까운 비교를 함.
    #       세 가지 요소를 종합 평가:
    #         - 밝기(luminance): 로컬 평균 μ 비교
    #         - 대비(contrast): 로컬 분산 σ 비교
    #         - 구조(structure): 공분산 σ12 비교
    #
    # 수식: SSIM = (2*μ1*μ2 + C1)(2*σ12 + C2) / ((μ1² + μ2² + C1)(σ1² + σ2² + C2))
    #       - μ: 가우시안 가중 로컬 평균
    #       - σ²: 가우시안 가중 로컬 분산
    #       - σ12: 가우시안 가중 공분산
    #       - C1, C2: 분모가 0이 되는 것을 방지하는 안정화 상수
    #         C = (K * MAX_VAL)^2, MAX_VAL = 10000
    #
    # 계산 과정:
    #   1. 11x11 가우시안 윈도우 생성 (σ=1.5)
    #   2. conv2d로 로컬 통계량(μ, σ², σ12) 계산
    #   3. SSIM 공식 적용 → 픽셀별 SSIM 맵 생성
    #   4. 전체 평균으로 스칼라 반환

    name = "ssim"

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size

        # 1D 가우시안 커널 생성
        gauss = torch.tensor(
            [math.exp(-((x - window_size // 2) ** 2) / (2.0 * sigma ** 2))
             for x in range(window_size)]
        )
        gauss = gauss / gauss.sum()

        # 1D → 2D 가우시안 윈도우 (외적)
        window_1d = gauss.unsqueeze(1)
        window_2d = window_1d.mm(window_1d.t()).unsqueeze(0).unsqueeze(0)

        # register_buffer: device 이동 시 함께 이동, 학습 파라미터는 아님
        self.register_buffer("_window_2d", window_2d)

    def forward(self, outputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 스케일 복원
        pred = outputs * 2000.0
        gt = target * 2000.0

        # 채널별 독립 계산 (groups=channel)
        channel = pred.size(1)
        window = self._window_2d.expand(channel, 1, -1, -1).to(pred.device)
        pad = self.window_size // 2

        # 가우시안 가중 로컬 평균 (μ)
        mu1 = F.conv2d(pred, window, padding=pad, groups=channel)
        mu2 = F.conv2d(gt, window, padding=pad, groups=channel)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        # 가우시안 가중 분산/공분산: Var(X) = E[X²] - E[X]²
        sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(gt * gt, window, padding=pad, groups=channel) - mu2_sq
        sigma12 = F.conv2d(pred * gt, window, padding=pad, groups=channel) - mu1_mu2

        # 안정화 상수
        max_val = 10000.0
        C1 = (0.01 * max_val) ** 2  # 10000.0
        C2 = (0.03 * max_val) ** 2  # 90000.0

        # SSIM 공식 적용
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return ssim_map.mean()


class SAM(nn.Module):
    """Spectral Angle Mapper (degrees) — 낮을수록 좋음."""

    # 출처: ACA-CRNet/predict_rice1.py 119~124행
    #
    # 개념: 각 픽셀의 스펙트럴 벡터(13개 밴드 값) 사이의 각도를 측정.
    #       밴드 간 비율(분광 특성)이 보존되었는지 평가하는 지표.
    #       밝기가 달라도 분광 패턴이 같으면 각도가 작음.
    #       → 토지 피복 분류 등 후속 작업의 품질을 간접 평가.
    #
    # 수식: cosθ = (pred · target) / (‖pred‖ × ‖target‖)
    #       SAM = mean(arccos(cosθ)) → degree 단위
    #
    # 계산 과정:
    #   1. 채널축(dim=1) 내적으로 두 벡터의 유사도 계산
    #   2. 각 벡터의 L2 노름으로 나눠서 코사인 유사도 산출
    #   3. arccos로 각도(라디안) 변환
    #   4. degree로 환산 후 전체 픽셀 평균

    name = "sam"

    def forward(self, outputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eps = 1e-8  # zero-division 방어

        # 채널축 내적: (B, C, H, W) → (B, H, W)
        dot = torch.sum(outputs * target, dim=1)

        # 각 벡터의 L2 노름
        norm_pred = torch.sqrt(torch.sum(outputs ** 2, dim=1))
        norm_gt = torch.sqrt(torch.sum(target ** 2, dim=1))

        # 코사인 유사도, [-1, 1] 클램핑 (부동소수점 오차 방어)
        cos_sim = dot / (norm_pred * norm_gt + eps)
        angle = torch.acos(torch.clamp(cos_sim, -1.0, 1.0))

        # 라디안 → 도 변환
        sam = torch.mean(angle) * (180.0 / math.pi)
        return sam
