from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class FDTDecompositionLoss(nn.Module):
    def __init__(
        self,
        *,
        common_weight: float = 1.0,
        comp_weight: float = 1.0,
        window_size: int = 11,
        sigma: float = 1.5,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.common_weight = common_weight
        self.comp_weight = comp_weight
        self.window_size = window_size
        self.eps = eps

        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        gauss = torch.exp(-(coords.square()) / (2.0 * sigma**2))
        gauss = gauss / gauss.sum()
        window = gauss[:, None] @ gauss[None, :]
        self.register_buffer("_window_2d", window.view(1, 1, window_size, window_size))

    def _feature_norm(self, feature: torch.Tensor) -> torch.Tensor:
        min_value = feature.amin(dim=(2, 3), keepdim=True)
        max_value = feature.amax(dim=(2, 3), keepdim=True)
        return (feature - min_value) / (max_value - min_value + self.eps)

    def _feature_ssim(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        first = self._feature_norm(first)
        second = self._feature_norm(second)

        channel = first.size(1)
        window = self._window_2d.to(device=first.device, dtype=first.dtype)
        window = window.expand(channel, 1, -1, -1)
        pad = self.window_size // 2

        mu1 = F.conv2d(first, window, padding=pad, groups=channel)
        mu2 = F.conv2d(second, window, padding=pad, groups=channel)

        mu1_sq = mu1.square()
        mu2_sq = mu2.square()
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(first * first, window, padding=pad, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(second * second, window, padding=pad, groups=channel) - mu2_sq
        sigma12 = F.conv2d(first * second, window, padding=pad, groups=channel) - mu1_mu2

        c1 = math.pow(0.01, 2)
        c2 = math.pow(0.03, 2)
        ssim_map = ((2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + self.eps
        )
        return ssim_map.mean()

    def _local_corr_squared(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> torch.Tensor:
        channel = first.size(1)
        window = self._window_2d.to(device=first.device, dtype=first.dtype)
        window = window.expand(channel, 1, -1, -1)
        pad = self.window_size // 2

        mu1 = F.conv2d(first, window, padding=pad, groups=channel)
        mu2 = F.conv2d(second, window, padding=pad, groups=channel)

        var1 = F.conv2d(first * first, window, padding=pad, groups=channel) - mu1.square()
        var2 = F.conv2d(second * second, window, padding=pad, groups=channel) - mu2.square()
        cov12 = F.conv2d(first * second, window, padding=pad, groups=channel) - mu1 * mu2

        var1 = var1.clamp_min(0.0)
        var2 = var2.clamp_min(0.0)
        corr12 = cov12 / torch.sqrt(var1 * var2 + self.eps)
        return corr12.square().mean()

    def forward(
        self,
        sar_com: torch.Tensor,
        cld_com: torch.Tensor,
        sar_comp: torch.Tensor,
        cld_comp: torch.Tensor,
    ) -> torch.Tensor:
        common_loss = 1.0 - self._feature_ssim(sar_com, cld_com)
        comp_loss = self._local_corr_squared(sar_comp, cld_comp)
        return self.common_weight * common_loss + self.comp_weight * comp_loss
