from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from modules.metrics import SSIM


def _check_same_shape(first: torch.Tensor, second: torch.Tensor) -> None:
    if first.shape != second.shape:
        raise ValueError(f"feature shapes must match: {first.shape} != {second.shape}")


class PatchSlicedWassersteinLoss(nn.Module):
    def __init__(
        self,
        *,
        num_projections: int = 128,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.num_projections = num_projections
        self.eps = eps

    def _projection_weight(self, feature: torch.Tensor) -> torch.Tensor:
        weight = torch.randn(
            self.num_projections,
            feature.shape[1],
            3,
            3,
            device=feature.device,
            dtype=torch.float32,
        )
        norm = weight.flatten(1).norm(dim=1).clamp_min(self.eps)
        return weight / norm.view(-1, 1, 1, 1)

    def _project(self, feature: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        feature = F.pad(feature, (1, 1, 1, 1), mode="replicate")
        return F.conv2d(feature, weight).flatten(2)

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        _check_same_shape(first, second)
        with torch.autocast(device_type=first.device.type, enabled=False):
            first = first.float()
            second = second.float()
            weight = self._projection_weight(first)
            first = self._project(first, weight).sort(dim=-1).values
            second = self._project(second, weight).sort(dim=-1).values
            return (first - second).abs().mean()


class FeatureUncorrelationLoss(nn.Module):
    def __init__(self, *, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps

    def _centered_corr_square(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
        *,
        dim: int,
    ) -> torch.Tensor:
        first = first - first.mean(dim=dim, keepdim=True)
        second = second - second.mean(dim=dim, keepdim=True)
        covariance = (first * second).mean(dim=dim)
        first_var = first.square().mean(dim=dim)
        second_var = second.square().mean(dim=dim)
        return covariance.square() / ((first_var + self.eps) * (second_var + self.eps))

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        _check_same_shape(first, second)
        with torch.autocast(device_type=first.device.type, enabled=False):
            first = first.float()
            second = second.float()
            channel_loss = self._centered_corr_square(
                first.flatten(2).transpose(1, 2),
                second.flatten(2).transpose(1, 2),
                dim=-1,
            )
            spatial_loss = self._centered_corr_square(
                first.flatten(2),
                second.flatten(2),
                dim=-1,
            )
            return channel_loss.mean() + spatial_loss.mean()


class FDTDecompositionLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.candidate_loss = nn.L1Loss()

    def forward(
        self,
        candidate: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        _check_same_shape(candidate, target)
        return self.candidate_loss(candidate, target)


class FDTCCALoss(nn.Module):
    def __init__(
        self,
        *,
        l1_weight: float = 1.0,
        ssim_weight: float = 0.5,
        candidate_loss_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.candidate_loss_weight = candidate_loss_weight
        self.l1_loss = nn.L1Loss()
        self.ssim = SSIM()
        self.candidate_loss = FDTDecompositionLoss()

    def forward(
        self,
        model_output: Any,
        target: torch.Tensor,
    ) -> torch.Tensor:
        prediction, candidate = self._split_model_output(model_output)
        _check_same_shape(prediction, target)
        loss = self.l1_weight * self.l1_loss(prediction, target)
        loss = loss + self.ssim_weight * (1.0 - self.ssim(prediction, target))
        if self.candidate_loss_weight == 0.0:
            return loss
        if candidate is None:
            raise ValueError("FDTCCALoss requires candidate output when candidate_loss_weight is non-zero")
        return loss + self.candidate_loss_weight * self.candidate_loss(candidate, target)

    def _split_model_output(
        self,
        model_output: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(model_output, torch.Tensor):
            return model_output, None
        if not isinstance(model_output, tuple) or len(model_output) < 2:
            raise TypeError("FDTCCALoss expects a tensor or FDT-CCA output tuple")
        prediction = model_output[0]
        candidate = model_output[1]
        if not isinstance(prediction, torch.Tensor):
            raise TypeError("FDT-CCA prediction output must be a tensor")
        if not isinstance(candidate, torch.Tensor):
            raise TypeError("FDT-CCA candidate output must be a tensor")
        return prediction, candidate
