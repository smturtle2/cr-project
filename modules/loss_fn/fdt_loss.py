from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def _check_same_shape(first: torch.Tensor, second: torch.Tensor) -> None:
    if first.shape != second.shape:
        raise ValueError(f"feature shapes must match: {first.shape} != {second.shape}")


class _SSIMLoss(nn.Module):
    def __init__(
        self,
        *,
        window_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 5.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.data_range = data_range
        self.eps = eps
        gauss = torch.tensor(
            [
                math.exp(-((x - window_size // 2) ** 2) / (2.0 * sigma**2))
                for x in range(window_size)
            ],
            dtype=torch.float32,
        )
        gauss = gauss / gauss.sum()
        window_1d = gauss.unsqueeze(1)
        window_2d = window_1d.mm(window_1d.t()).unsqueeze(0).unsqueeze(0)
        self.register_buffer("_window_2d", window_2d)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        _check_same_shape(prediction, target)
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            prediction = prediction.float()
            target = target.float()
            channel = prediction.size(1)
            window = self._window_2d.to(
                device=prediction.device,
                dtype=prediction.dtype,
            ).expand(channel, 1, -1, -1)
            pad = self.window_size // 2

            mu1 = F.conv2d(prediction, window, padding=pad, groups=channel)
            mu2 = F.conv2d(target, window, padding=pad, groups=channel)
            mu1_sq = mu1.square()
            mu2_sq = mu2.square()
            mu1_mu2 = mu1 * mu2

            sigma1_sq = (
                F.conv2d(prediction.square(), window, padding=pad, groups=channel)
                - mu1_sq
            ).clamp_min(0.0)
            sigma2_sq = (
                F.conv2d(target.square(), window, padding=pad, groups=channel)
                - mu2_sq
            ).clamp_min(0.0)
            sigma12 = (
                F.conv2d(prediction * target, window, padding=pad, groups=channel)
                - mu1_mu2
            )

            c1 = (0.01 * self.data_range) ** 2
            c2 = (0.03 * self.data_range) ** 2
            numerator = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
            denominator = ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
            ssim_map = numerator / denominator.clamp_min(self.eps)
            ssim_map = torch.nan_to_num(ssim_map, nan=-1.0, posinf=1.0, neginf=-1.0)
            return 1.0 - ssim_map.clamp(-1.0, 1.0).mean()


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
        self.ssim_loss = _SSIMLoss()
        self.candidate_loss = nn.L1Loss()

    def forward(
        self,
        model_output: Any,
        target: torch.Tensor,
    ) -> torch.Tensor:
        prediction, candidate = self._split_model_output(model_output)
        _check_same_shape(prediction, target)
        loss = self.l1_weight * self.l1_loss(prediction, target)
        loss = loss + self.ssim_weight * self.ssim_loss(prediction, target)
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
