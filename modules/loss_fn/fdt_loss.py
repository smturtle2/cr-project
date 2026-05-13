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
    def _tokens(feature: torch.Tensor) -> torch.Tensor:
        return feature.flatten(2).transpose(1, 2).float()

    def _joint_pc1_maps(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, height, width = first.shape
        num_positions = height * width
        with torch.autocast(device_type=first.device.type, enabled=False):
            first_tokens = self._tokens(first)
            second_tokens = self._tokens(second)
            merged = torch.cat((first_tokens, second_tokens), dim=1)
            centered = merged - merged.mean(dim=1, keepdim=True)

        with torch.no_grad(), torch.autocast(device_type=first.device.type, enabled=False):
            basis_source = centered.detach().float()
            covariance = basis_source.transpose(1, 2) @ basis_source
            _, eigenvectors = torch.linalg.eigh(covariance)
            pc1 = eigenvectors[:, :, -1]

        with torch.autocast(device_type=first.device.type, enabled=False):
            projected = torch.bmm(centered.float(), pc1.unsqueeze(-1)).squeeze(-1)
        first_map = projected[:, :num_positions].reshape(batch_size, height, width)
        second_map = projected[:, num_positions:].reshape(batch_size, height, width)
        return first_map, second_map

    def _map_ncc(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> torch.Tensor:
        first = first.flatten(1)
        second = second.flatten(1)
        first = first - first.mean(dim=1, keepdim=True)
        second = second - second.mean(dim=1, keepdim=True)

        numerator = (first * second).sum(dim=1)
        denominator = torch.sqrt(
            first.square().sum(dim=1) * second.square().sum(dim=1) + self.eps
        )
        return numerator / denominator

    def _map_ccc(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> torch.Tensor:
        first = first.flatten(1)
        second = second.flatten(1)
        first = first - first.mean(dim=1, keepdim=True)
        second = second - second.mean(dim=1, keepdim=True)

        cov = (first * second).mean(dim=1)
        first_var = first.square().mean(dim=1)
        second_var = second.square().mean(dim=1)
        return 2.0 * cov / (first_var + second_var + self.eps)

    def _common_alignment_loss(
        self,
        sar_com: torch.Tensor,
        cld_com: torch.Tensor,
    ) -> torch.Tensor:
        sar_map, cld_map = self._joint_pc1_maps(sar_com, cld_com)
        return 1.0 - self._map_ccc(sar_map, cld_map).mean()

    def _comp_decorrelation_loss(
        self,
        sar_comp: torch.Tensor,
        cld_comp: torch.Tensor,
    ) -> torch.Tensor:
        sar_map, cld_map = self._joint_pc1_maps(sar_comp, cld_comp)
        return self._map_ccc(sar_map, cld_map).square().mean()

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
