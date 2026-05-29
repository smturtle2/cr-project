from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def _check_same_shape(first: torch.Tensor, second: torch.Tensor) -> None:
    if first.shape != second.shape:
        raise ValueError(f"feature shapes must match: {first.shape} != {second.shape}")


class _CharbonnierLoss(nn.Module):
    def __init__(
        self,
        *,
        eps: float = 1e-3,
    ) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        _check_same_shape(prediction, target)
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            prediction = prediction.float()
            target = target.float()
            return torch.sqrt((prediction - target).square() + self.eps**2).mean()


class _LoGLoss(nn.Module):
    def __init__(
        self,
        *,
        kernel_size: int = 5,
        sigma: float = 1.0,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if sigma <= 0.0:
            raise ValueError("sigma must be positive")

        offset = kernel_size // 2
        coords = torch.arange(-offset, offset + 1, dtype=torch.float32)
        gaussian_1d = torch.exp(-(coords.square()) / (2.0 * sigma**2))
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        gaussian_2d = gaussian_1d[:, None] * gaussian_1d[None, :]
        laplacian = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        self.register_buffer(
            "_gaussian_kernel",
            gaussian_2d.view(1, 1, kernel_size, kernel_size),
        )
        self.register_buffer("_laplacian_kernel", laplacian.view(1, 1, 3, 3))

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        _check_same_shape(prediction, target)
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            prediction = prediction.float()
            target = target.float()
            return F.l1_loss(self._log(prediction), self._log(target))

    def _log(self, image: torch.Tensor) -> torch.Tensor:
        channel = image.size(1)
        gaussian_kernel = self._gaussian_kernel.to(
            device=image.device,
            dtype=image.dtype,
        ).expand(
            channel,
            1,
            -1,
            -1,
        )
        laplacian_kernel = self._laplacian_kernel.to(
            device=image.device,
            dtype=image.dtype,
        ).expand(
            channel,
            1,
            3,
            3,
        )
        gaussian_pad = self._gaussian_kernel.size(-1) // 2
        image = F.pad(
            image,
            (gaussian_pad, gaussian_pad, gaussian_pad, gaussian_pad),
            mode="replicate",
        )
        image = F.conv2d(image, gaussian_kernel, groups=channel)
        image = F.pad(image, (1, 1, 1, 1), mode="replicate")
        return F.conv2d(image, laplacian_kernel, groups=channel)


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


class _SAMLoss(nn.Module):
    def __init__(self, *, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        _check_same_shape(prediction, target)
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            prediction = prediction.float()
            target = target.float()
            cosine = F.cosine_similarity(prediction, target, dim=1, eps=self.eps)
            return torch.acos(cosine.clamp(-1.0 + self.eps, 1.0 - self.eps)).mean()


class FDTCCALoss(nn.Module):
    def __init__(
        self,
        *,
        charbonnier_weight: float = 1.0,
        candidate_loss_weight: float = 0.1,
        log_weight: float = 0.05,
        sam_weight: float = 0.02,
        ssim_weight: float = 0.1,
        laplacian_weight: float | None = None,
    ) -> None:
        super().__init__()
        if laplacian_weight is not None:
            log_weight = laplacian_weight
        self.charbonnier_weight = charbonnier_weight
        self.candidate_loss_weight = candidate_loss_weight
        self.log_weight = log_weight
        self.sam_weight = sam_weight
        self.ssim_weight = ssim_weight
        self.charbonnier_loss = _CharbonnierLoss()
        self.candidate_loss = _CharbonnierLoss()
        self.log_loss = _LoGLoss()
        self.laplacian_loss = self.log_loss
        self.sam_loss = _SAMLoss()
        self.ssim_loss = _SSIMLoss()

    def forward(
        self,
        model_output: Any,
        target: torch.Tensor,
    ) -> torch.Tensor:
        prediction, candidate = self._split_model_output(model_output)
        _check_same_shape(prediction, target)
        loss = self.charbonnier_weight * self.charbonnier_loss(prediction, target)
        loss = loss + self.log_weight * self.log_loss(prediction, target)
        loss = loss + self.sam_weight * self.sam_loss(prediction, target)
        loss = loss + self.ssim_weight * self.ssim_loss(prediction, target)
        if self.candidate_loss_weight != 0.0 and candidate is None:
            raise ValueError("FDTCCALoss requires candidate output when candidate_loss_weight is non-zero")
        if candidate is not None:
            _check_same_shape(candidate, target)
            loss = loss + self.candidate_loss_weight * self.candidate_loss(candidate, target)
        return loss

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
