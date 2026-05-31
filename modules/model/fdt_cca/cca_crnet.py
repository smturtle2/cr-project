from __future__ import annotations

import torch
import torch.nn as nn

from ..fdt.ACA_CRNet import ACA_CRNet, DefaultConAttn
from ..fdt.fdt import Residual3x3Block


class CCAMask(nn.Module):
    def __init__(
        self,
        cloud_channels: int,
        mask_channels: int,
    ):
        super().__init__()
        self.body = nn.Sequential(
            Residual3x3Block(cloud_channels),
            Residual3x3Block(cloud_channels),
        )
        self.mask_head = nn.Conv2d(cloud_channels, mask_channels, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, cld_cloud: torch.Tensor) -> torch.Tensor:
        return self.activation(self.mask_head(self.body(cld_cloud)))


class CCA_CRNet(ACA_CRNet):
    def __init__(
        self,
        out_channels: int,
        alpha: float = 0.1,
        num_layers: int = 16,
        feature_sizes: int = 256,
        cloud_channels: int | None = None,
        cca_layers: int = 1,
        num_heads: int = 4,
        ca=DefaultConAttn,
        ca_kwargs=None,
        detail_blocks: int = 2,
    ):
        del cca_layers, num_heads, detail_blocks
        super().__init__(
            out_channels=out_channels,
            alpha=alpha,
            num_layers=num_layers,
            feature_sizes=feature_sizes,
            ca=ca,
            ca_kwargs=ca_kwargs,
            mode="direct",
        )
        cloud_channels = feature_sizes // 2 if cloud_channels is None else cloud_channels
        candidate_head = self.net[-1]
        self.net = nn.ModuleList(self.net[:-1])
        self.candidate_head = candidate_head
        self.cca = CCAMask(cloud_channels, out_channels)

    def forward(
        self,
        feature: torch.Tensor,
        cld_cloud: torch.Tensor,
        cloudy: torch.Tensor,
        *,
        return_candidate: bool = False,
        return_mask: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        z = feature
        for layer in self.net:
            z = layer(z)

        candidate = self.candidate_head(z).clamp(0.0, 5.0)
        mask = self.cca(cld_cloud)
        prediction = cloudy * (1.0 - mask) + candidate * mask
        output = {"prediction": prediction}
        if return_candidate:
            output["candidate"] = candidate
        if return_mask:
            output["mask"] = mask
        if return_candidate or return_mask:
            return output
        return prediction
