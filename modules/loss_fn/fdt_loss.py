from __future__ import annotations

import torch
from torch import nn


class FDTDecompositionLoss(nn.Module):
    def __init__(
        self,
        *,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps

    def _centered_ccc(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> torch.Tensor:
        first = first - first.mean(dim=-1, keepdim=True)
        second = second - second.mean(dim=-1, keepdim=True)

        cov = (first * second).mean(dim=-1)
        first_var = first.square().mean(dim=-1)
        second_var = second.square().mean(dim=-1)
        return 2.0 * cov / (first_var + second_var + self.eps)

    def _channel_axis_ccc(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> torch.Tensor:
        first = first.flatten(2).transpose(1, 2)
        second = second.flatten(2).transpose(1, 2)
        return self._centered_ccc(first, second).mean(dim=1)

    def _spatial_axis_ccc(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> torch.Tensor:
        first = first.flatten(2)
        second = second.flatten(2)
        return self._centered_ccc(first, second).mean(dim=1)

    def _feature_ccc(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> torch.Tensor:
        with torch.autocast(device_type=first.device.type, enabled=False):
            first = first.float()
            second = second.float()
            channel_score = self._channel_axis_ccc(first, second)
            spatial_score = self._spatial_axis_ccc(first, second)
            return 0.5 * (channel_score + spatial_score)

    def _common_alignment_loss(
        self,
        sar_com: torch.Tensor,
        cld_com: torch.Tensor,
    ) -> torch.Tensor:
        return 1.0 - self._feature_ccc(sar_com, cld_com).mean()

    def _comp_decorrelation_loss(
        self,
        sar_comp: torch.Tensor,
        cld_comp: torch.Tensor,
    ) -> torch.Tensor:
        return self._feature_ccc(sar_comp, cld_comp).square().mean()

    def forward(
        self,
        sar_com: torch.Tensor,
        cld_com: torch.Tensor,
        sar_comp: torch.Tensor,
        cld_comp: torch.Tensor,
    ) -> torch.Tensor:
        common_loss = self._common_alignment_loss(sar_com, cld_com)
        comp_loss = self._comp_decorrelation_loss(sar_comp, cld_comp)
        return common_loss + comp_loss
