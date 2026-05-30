from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def _check_same_shape(first: torch.Tensor, second: torch.Tensor) -> None:
    if first.shape != second.shape:
        raise ValueError(f"feature shapes must match: {first.shape} != {second.shape}")


class FDTCCALoss(nn.Module):
    """Output L1 + SSIM loss adapted to this project's /2000 tensor scale."""

    def __init__(
        self,
        *,
        l1_weight: float = 0.9,
        ssim_weight: float = 0.1,
        input_scale: float = 5.0,
    ) -> None:
        super().__init__()
        if input_scale <= 0.0:
            raise ValueError("input_scale must be positive")

        self.l1_weight = float(l1_weight)
        self.ssim_weight = float(ssim_weight)
        self.input_scale = float(input_scale)
        self.ssim = _GaussianSSIM()

    def forward(
        self,
        model_output: Any,
        target: torch.Tensor,
    ) -> torch.Tensor:
        prediction = self._prediction_from_output(model_output)
        return self.output_loss(prediction, target)

    def _prediction_from_output(
        self,
        model_output: Any,
    ) -> torch.Tensor:
        if isinstance(model_output, torch.Tensor):
            return model_output
        if not isinstance(model_output, tuple) or len(model_output) < 1:
            raise TypeError("FDTCCALoss expects a tensor or FDT-CCA output tuple")
        prediction = model_output[0]
        if not isinstance(prediction, torch.Tensor):
            raise TypeError("FDT-CCA prediction output must be a tensor")
        return prediction

    def output_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        _check_same_shape(prediction, target)
        prediction = self._normalize(prediction)
        target = self._normalize(target)
        return (
            self.l1_weight * F.l1_loss(prediction, target)
            + self.ssim_weight * (1.0 - self.ssim(prediction, target))
        )

    def _normalize(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        # Local SEN12MS-CR tensors use reflectance / 2000; normalize to /10000.
        return tensor.float() / self.input_scale

    def ssim_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        _check_same_shape(prediction, target)
        return 1.0 - self.ssim(self._normalize(prediction), self._normalize(target))


def make_fdt_cca_loss_fn(
    *,
    l1_weight: float = 0.9,
    ssim_weight: float = 0.1,
    input_scale: float = 5.0,
) -> Callable[[Any, Mapping[str, torch.Tensor]], torch.Tensor]:
    criterion = FDTCCALoss(
        l1_weight=l1_weight,
        ssim_weight=ssim_weight,
        input_scale=input_scale,
    )

    def loss_fn(
        model_output: Any,
        batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        return criterion(model_output, batch["target"])

    return loss_fn


class _GaussianSSIM(nn.Module):
    def __init__(
        self,
        *,
        window_size: int = 11,
        sigma: float = 1.5,
        eps: float = 1.6e-9,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.eps = eps
        self.register_buffer("_window", self._create_window(window_size, sigma))

    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
    ) -> torch.Tensor:
        _check_same_shape(img1, img2)
        with torch.autocast(device_type=img1.device.type, enabled=False):
            img1 = img1.float()
            img2 = img2.float()
            channel = img1.size(1)
            window = self._window.to(device=img1.device, dtype=img1.dtype).expand(
                channel,
                1,
                self.window_size,
                self.window_size,
            )
            return self._ssim(img1, img2, window, channel)

    def _create_window(
        self,
        window_size: int,
        sigma: float,
    ) -> torch.Tensor:
        gauss = torch.tensor(
            [
                math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
                for x in range(window_size)
            ],
            dtype=torch.float32,
        )
        gauss = gauss / gauss.sum()
        window_1d = gauss.unsqueeze(1)
        window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
        return window_2d

    def _ssim(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        window: torch.Tensor,
        channel: int,
    ) -> torch.Tensor:
        mu1 = self._conv(img1, window, channel)
        mu2 = self._conv(img2, window, channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (self._conv(img1 * img1, window, channel) - mu1_sq).clamp_min(0.0)
        sigma2_sq = (self._conv(img2 * img2, window, channel) - mu2_sq).clamp_min(0.0)
        sigma12 = self._conv(img1 * img2, window, channel) - mu1_mu2

        c1 = 0.01**2
        c2 = 0.03**2
        numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
        denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        ssim_map = numerator / denominator.clamp_min(self.eps)
        ssim_map = torch.nan_to_num(ssim_map, nan=-1.0, posinf=1.0, neginf=-1.0)

        return ssim_map.clamp(-1.0, 1.0).mean()

    def _conv(
        self,
        image: torch.Tensor,
        window: torch.Tensor,
        channel: int,
    ) -> torch.Tensor:
        return F.conv2d(
            image,
            window,
            padding=self.window_size // 2,
            groups=channel,
        ).float()
