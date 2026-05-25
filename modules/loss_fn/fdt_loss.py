from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


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

    def _centered_corr(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
        *,
        dim: int,
    ) -> torch.Tensor:
        first = first - first.mean(dim=dim, keepdim=True)
        second = second - second.mean(dim=dim, keepdim=True)
        covariance = (first * second).mean(dim=dim)
        first_std = first.square().mean(dim=dim).sqrt()
        second_std = second.square().mean(dim=dim).sqrt()
        return covariance / (first_std * second_std + self.eps)

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        _check_same_shape(first, second)
        with torch.autocast(device_type=first.device.type, enabled=False):
            first = first.float()
            second = second.float()
            channel_corr = self._centered_corr(
                first.flatten(2).transpose(1, 2),
                second.flatten(2).transpose(1, 2),
                dim=-1,
            )
            spatial_corr = self._centered_corr(
                first.flatten(2),
                second.flatten(2),
                dim=-1,
            )
            return channel_corr.square().mean() + spatial_corr.square().mean()


class FDTDecompositionLoss(nn.Module):
    def __init__(
        self,
        *,
        num_projections: int = 128,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.common_loss = PatchSlicedWassersteinLoss(
            num_projections=num_projections,
            eps=eps,
        )
        self.comp_loss = FeatureUncorrelationLoss(eps=eps)

    def forward(
        self,
        sar_com: torch.Tensor,
        cld_com: torch.Tensor,
        sar_comp: torch.Tensor,
        cld_comp: torch.Tensor,
    ) -> torch.Tensor:
        common_loss = self.common_loss(sar_com, cld_com)
        comp_loss = self.comp_loss(sar_comp, cld_comp)
        return common_loss + comp_loss
