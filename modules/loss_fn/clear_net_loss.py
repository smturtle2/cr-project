from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _check_same_shape(first: torch.Tensor, second: torch.Tensor) -> None:
    if first.shape != second.shape:
        raise ValueError(f"feature shapes must match: {first.shape} != {second.shape}")


class CLEAR_NetLoss(nn.Module):
    def __init__(
        self,
        *,
        ssim_weight: float = 0.1,
        prediction_weight: float = 1.0,
        candidate_weight: float = 0.5,
        aux_weight: float = 0.2,
        route_balance_weight: float = 0.002,
        mask_reconstruction_weight: float = 0.5,
        mask_reconstruction_max_ratio: float = 4.0,
        data_range: float = 5.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.ssim_weight = float(ssim_weight)
        self.prediction_weight = float(prediction_weight)
        self.candidate_weight = float(candidate_weight)
        self.aux_weight = float(aux_weight)
        self.route_balance_weight = float(route_balance_weight)
        self.mask_reconstruction_weight = float(mask_reconstruction_weight)
        self.mask_reconstruction_max_ratio = float(mask_reconstruction_max_ratio)
        self.eps = float(eps)
        self.ssim = GaussianSSIM(data_range=data_range)

    def forward(
        self,
        model_output: Any,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(model_output, Mapping):
            prediction = model_output["prediction"]
            candidate = model_output.get("candidate")
            aux_prediction = model_output.get("aux_clear")
            mask = model_output.get("mask")
            route_weights = model_output.get("route_weights")
        else:
            prediction = model_output
            candidate = None
            aux_prediction = None
            mask = None
            route_weights = None

        _check_same_shape(prediction, target)
        prediction = prediction.float()
        target = target.float()
        mask_weight = None
        if mask is not None and self.mask_reconstruction_weight != 0.0:
            mask_weight = mask.detach().float().clamp(0.0, 1.0)
            min_mask_sum = (
                mask_weight.numel()
                * self.mask_reconstruction_weight
                / (self.mask_reconstruction_max_ratio - 1.0)
            )
            mask_weight = mask_weight / mask_weight.sum().clamp_min(min_mask_sum)

        loss = self.prediction_weight * self.reconstruction_loss(
            prediction,
            target,
            mask_weight=mask_weight,
        )
        if candidate is not None:
            loss = loss + self.candidate_weight * self.reconstruction_loss(
                candidate,
                target,
                mask_weight=mask_weight,
            )
        if aux_prediction is not None:
            loss = loss + self.aux_weight * self.reconstruction_loss(
                aux_prediction,
                target,
                mask_weight=mask_weight,
            )
        if route_weights is not None:
            loss = loss + self.route_balance_weight * self.route_balance_loss(route_weights)
        return loss

    def output_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        mask_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.reconstruction_loss(prediction, target, mask_weight=mask_weight)

    def aux_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        mask_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.reconstruction_loss(prediction, target, mask_weight=mask_weight)

    def route_balance_loss(self, route_weights: torch.Tensor) -> torch.Tensor:
        route_weights = route_weights.float()
        route_count = route_weights.size(1)
        router_prob_per_route = route_weights.mean(dim=(0, 2, 3))
        selected_routes = route_weights.argmax(dim=1)
        route_usage = F.one_hot(selected_routes, num_classes=route_count).float()
        route_usage = route_usage.mean(dim=(0, 1, 2)).detach()
        return route_count * (route_usage * router_prob_per_route).sum()

    def reconstruction_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        mask_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _check_same_shape(prediction, target)
        prediction = prediction.float()
        target = target.float()
        error = (prediction - target).abs()
        loss = error.mean()
        if mask_weight is not None and self.mask_reconstruction_weight != 0.0:
            loss = loss + self.mask_reconstruction_weight * self.mask_l1_loss(
                error,
                mask_weight,
            )
        if self.ssim_weight != 0.0:
            loss = loss + self.ssim_weight * self.ssim_loss(prediction, target)
        return loss

    def mask_l1_loss(
        self,
        error: torch.Tensor,
        mask_weight: torch.Tensor,
    ) -> torch.Tensor:
        return (error * mask_weight).sum()

    def ssim_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _check_same_shape(prediction, target)
        return 1.0 - self.ssim(prediction.float(), target.float())


def make_clear_net_loss_fn(
    *,
    ssim_weight: float = 0.1,
    prediction_weight: float = 1.0,
    candidate_weight: float = 0.5,
    aux_weight: float = 0.2,
    route_balance_weight: float = 0.002,
    mask_reconstruction_weight: float = 0.5,
    mask_reconstruction_max_ratio: float = 4.0,
    data_range: float = 5.0,
) -> Callable[[Any, Mapping[str, torch.Tensor]], torch.Tensor]:
    criterion = CLEAR_NetLoss(
        ssim_weight=ssim_weight,
        prediction_weight=prediction_weight,
        candidate_weight=candidate_weight,
        aux_weight=aux_weight,
        route_balance_weight=route_balance_weight,
        mask_reconstruction_weight=mask_reconstruction_weight,
        mask_reconstruction_max_ratio=mask_reconstruction_max_ratio,
        data_range=data_range,
    )

    def loss_fn(model_output: Any, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return criterion(model_output, batch["target"])

    return loss_fn


class GaussianSSIM(nn.Module):
    def __init__(
        self,
        *,
        window_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 5.0,
        eps: float = 1.6e-9,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.data_range = float(data_range)
        self.eps = eps
        self.register_buffer("_window", self._create_window(window_size, sigma))

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
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

    def _create_window(self, window_size: int, sigma: float) -> torch.Tensor:
        gauss = torch.tensor(
            [
                math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
                for x in range(window_size)
            ],
            dtype=torch.float32,
        )
        gauss = gauss / gauss.sum()
        window_1d = gauss.unsqueeze(1)
        return window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)

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

        c1 = (0.01 * self.data_range) ** 2
        c2 = (0.03 * self.data_range) ** 2
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
