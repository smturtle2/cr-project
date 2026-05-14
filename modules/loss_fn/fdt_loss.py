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
        # [B, N] -> [B, N], where N = C*H*W
        first = first - first.mean(dim=1, keepdim=True)
        second = second - second.mean(dim=1, keepdim=True)
        # [B, N] -> [B]
        numerator = 2.0 * (first * second).mean(dim=1)
        denominator = first.square().mean(dim=1) + second.square().mean(dim=1)
        return numerator / (denominator + self.eps)

    def _feature_score(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> torch.Tensor:
        with torch.autocast(device_type=first.device.type, enabled=False):
            # [B, C, H, W] -> [B, C*H*W]
            first = first.float().flatten(1)
            second = second.float().flatten(1)
            return self._centered_ccc(first, second)

    def _common_alignment_loss(
        self,
        sar_com: torch.Tensor,
        cld_com: torch.Tensor,
    ) -> torch.Tensor:
        return (1.0 - self._feature_score(sar_com, cld_com).mean()) / 2.0

    def _comp_decorrelation_loss(
        self,
        sar_comp: torch.Tensor,
        cld_comp: torch.Tensor,
    ) -> torch.Tensor:
        return self._feature_score(sar_comp, cld_comp).square().mean()

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
