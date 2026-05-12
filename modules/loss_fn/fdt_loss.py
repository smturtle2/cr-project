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

    @staticmethod
    def _flatten_spatial(feature: torch.Tensor) -> torch.Tensor:
        return feature.flatten(2).float()

    def _spatial_channel_similarity(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> torch.Tensor:
        first = self._flatten_spatial(first)
        second = self._flatten_spatial(second)
        first = first - first.mean(dim=1, keepdim=True)
        second = second - second.mean(dim=1, keepdim=True)

        numerator = (first * second).sum(dim=1)
        denominator = torch.sqrt(
            first.square().sum(dim=1) * second.square().sum(dim=1) + self.eps
        )
        return numerator / denominator

    def _common_alignment_loss(
        self,
        sar_com: torch.Tensor,
        cld_com: torch.Tensor,
    ) -> torch.Tensor:
        similarity = self._spatial_channel_similarity(sar_com, cld_com)
        return 1.0 - similarity.mean()

    def _comp_decorrelation_loss(
        self,
        sar_comp: torch.Tensor,
        cld_comp: torch.Tensor,
    ) -> torch.Tensor:
        similarity = self._spatial_channel_similarity(sar_comp, cld_comp)
        return similarity.square().mean()

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
