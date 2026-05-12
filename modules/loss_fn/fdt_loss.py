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
    def _flatten_tokens(feature: torch.Tensor) -> torch.Tensor:
        return feature.flatten(2).transpose(1, 2).float()

    def _cross_channel_corr(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> torch.Tensor:
        first = self._flatten_tokens(first)
        second = self._flatten_tokens(second)
        first = first - first.mean(dim=1, keepdim=True)
        second = second - second.mean(dim=1, keepdim=True)

        first = first / torch.sqrt(first.square().sum(dim=1, keepdim=True) + self.eps)
        second = second / torch.sqrt(second.square().sum(dim=1, keepdim=True) + self.eps)
        return torch.bmm(first.transpose(1, 2), second)

    def _common_alignment_loss(
        self,
        sar_com: torch.Tensor,
        cld_com: torch.Tensor,
    ) -> torch.Tensor:
        corr = self._cross_channel_corr(sar_com, cld_com)
        return 1.0 - corr.diagonal(dim1=1, dim2=2).mean()

    def _comp_decorrelation_loss(
        self,
        sar_comp: torch.Tensor,
        cld_comp: torch.Tensor,
    ) -> torch.Tensor:
        corr = self._cross_channel_corr(sar_comp, cld_comp)
        return corr.square().mean()

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
