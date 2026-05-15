from __future__ import annotations

import torch
from torch import nn


class FDTDecompositionLoss(nn.Module):
    def __init__(
        self,
        *,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.eps = eps

    def _centered_ccc(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
        *,
        dim: int,
    ) -> torch.Tensor:
        first = first - first.mean(dim=dim, keepdim=True)
        second = second - second.mean(dim=dim, keepdim=True)
        numerator = 2.0 * (first * second).mean(dim=dim)
        denominator = first.square().mean(dim=dim) + second.square().mean(dim=dim)
        return numerator / (denominator + self.eps)

    def _axis_ccc_scores(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type=first.device.type, enabled=False):
            first = first.float()
            second = second.float()
            # [B, C, H, W] -> [B, H*W, C] -> [B, H*W]
            channel_scores = self._centered_ccc(
                first.flatten(2).transpose(1, 2),
                second.flatten(2).transpose(1, 2),
                dim=-1,
            )
            # [B, C, H, W] -> [B, C, H*W] -> [B, C]
            spatial_scores = self._centered_ccc(
                first.flatten(2),
                second.flatten(2),
                dim=-1,
            )
            return channel_scores, spatial_scores

    def _feature_scores(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        channel_scores, spatial_scores = self._axis_ccc_scores(first, second)
        return channel_scores.mean(dim=1), spatial_scores.mean(dim=1)

    def _common_alignment_loss(
        self,
        sar_com: torch.Tensor,
        cld_com: torch.Tensor,
    ) -> torch.Tensor:
        channel_score, spatial_score = self._feature_scores(sar_com, cld_com)
        return ((1.0 - channel_score) + (1.0 - spatial_score)).mean()

    def _comp_decorrelation_loss(
        self,
        sar_comp: torch.Tensor,
        cld_comp: torch.Tensor,
    ) -> torch.Tensor:
        channel_scores, spatial_scores = self._axis_ccc_scores(sar_comp, cld_comp)
        return channel_scores.square().mean() + spatial_scores.square().mean()

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
