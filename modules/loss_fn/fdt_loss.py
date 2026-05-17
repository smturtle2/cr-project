from __future__ import annotations

import torch
from torch import nn


class FDTDecompositionLoss(nn.Module):
    def __init__(
        self,
        *,
        common_weight: float = 1.0,
        comp_weight: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.common_weight = common_weight
        self.comp_weight = comp_weight
        self.eps = eps

    def _minmax_norm(self, feature: torch.Tensor) -> torch.Tensor:
        flattened = feature.flatten(1)
        min_value = flattened.min(dim=1).values.view(-1, 1, 1, 1)
        max_value = flattened.max(dim=1).values.view(-1, 1, 1, 1)
        return (feature - min_value) / (max_value - min_value + self.eps)

    def _contrast_similarity(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> torch.Tensor:
        first = self._minmax_norm(first)
        second = self._minmax_norm(second)
        first_centered = first - first.mean(dim=(1, 2, 3), keepdim=True)
        second_centered = second - second.mean(dim=(1, 2, 3), keepdim=True)
        covariance = (first_centered * second_centered).mean(dim=(1, 2, 3))
        first_var = first_centered.square().mean(dim=(1, 2, 3))
        second_var = second_centered.square().mean(dim=(1, 2, 3))
        return (2.0 * covariance + self.eps) / (first_var + second_var + self.eps)

    def _ncc(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        first_centered = first - first.mean(dim=(1, 2, 3), keepdim=True)
        second_centered = second - second.mean(dim=(1, 2, 3), keepdim=True)
        numerator = (first_centered * second_centered).sum(dim=(1, 2, 3))
        first_norm = first_centered.square().sum(dim=(1, 2, 3)).sqrt()
        second_norm = second_centered.square().sum(dim=(1, 2, 3)).sqrt()
        return numerator / (first_norm * second_norm + self.eps)

    def forward(
        self,
        sar_com: torch.Tensor,
        cld_com: torch.Tensor,
        sar_comp: torch.Tensor,
        cld_comp: torch.Tensor,
    ) -> torch.Tensor:
        common_loss = 1.0 - self._contrast_similarity(sar_com, cld_com).mean()
        comp_loss = self._ncc(sar_comp, cld_comp).abs().mean()
        return self.common_weight * common_loss + self.comp_weight * comp_loss
