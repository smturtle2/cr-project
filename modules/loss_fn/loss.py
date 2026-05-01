import torch
import torch.nn as nn


class CloudAdaptiveLoss(nn.Module):
    """Cloud-Adaptive CARL Loss.

    CAFM이 생성한 연속 밀도맵(0~1)을 활용하여
    구름 영역에 더 강한 복원 페널티를 부여하는 손실 함수.

    수식:
        L = mean((1 - d) * |pred - cloudy|        (맑은 영역: 입력 보존)
              +   d * α * |pred - target|)         (구름 영역: GT 복원 강조)
            + mean(|pred - target|)                (전체 복원 기본항)

    출처: ACA-CRNet_CAFM/losses.py → CloudAdaptiveLoss
    """

    def __init__(self, model: nn.Module, alpha: float = 2.0):
        super().__init__()
        self.model = model
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        density = self.model.last_density   # (B, 1, H, W) or None
        cloudy = self.model.last_cloudy     # (B, 13, H, W)

        global_l1 = torch.mean(torch.abs(pred - target))

        if density is not None and cloudy is not None:
            clear_loss = (1 - density) * torch.abs(pred - cloudy)
            cloud_loss = density * self.alpha * torch.abs(pred - target)
            adaptive = torch.mean(clear_loss + cloud_loss)
            return adaptive + global_l1
        else:
            return global_l1


class SimpleMSELoss(nn.Module):
    """단순 MSE Loss — CAFM 비활성 시 baseline 용."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean((pred - target) ** 2)


class SimpleL1Loss(nn.Module):
    """단순 L1 Loss — MSE 대안.

    구름 제거 분야에서 MSE보다 선호되는 표준 loss.
    MSE 대비 blur가 적고 엣지 보존이 나음 (DSen2-CR, CERMF-Net 등 채택).
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(pred - target))
