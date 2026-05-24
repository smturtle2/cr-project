from __future__ import annotations

import torch
import torch.nn as nn

from ..fdt.ACA_CRNet import ACA_CRNet, DefaultConAttn
from ..fdt.fdt import Residual3x3Block


class CCAMask(nn.Module):
    def __init__(self, comp_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            Residual3x3Block(comp_channels),
            Residual3x3Block(comp_channels),
            nn.Conv2d(comp_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, cld_comp: torch.Tensor) -> torch.Tensor:
        return self.net(cld_comp)


class CCA_CRNet(ACA_CRNet):
    def __init__(
        self,
        out_channels: int,
        alpha: float = 0.1,
        num_layers: int = 16,
        feature_sizes: int = 256,
        comp_channels: int | None = None,
        cca_layers: int = 1,
        num_heads: int = 4,
        ca=DefaultConAttn,
        ca_kwargs=None,
        detail_blocks: int = 2,
    ):
        del cca_layers, num_heads
        super().__init__(
            out_channels=out_channels,
            alpha=alpha,
            num_layers=num_layers,
            feature_sizes=feature_sizes,
            ca=ca,
            ca_kwargs=ca_kwargs,
            mode="direct",
        )
        comp_channels = feature_sizes // 2 if comp_channels is None else comp_channels
        low_head = self.net[-1]
        self.net = nn.ModuleList(self.net[:-1])
        self.low_head = low_head
        self.high_refine = nn.Sequential(
            *[Residual3x3Block(feature_sizes) for _ in range(detail_blocks)]
        )
        self.high_head = nn.Conv2d(
            feature_sizes,
            out_channels,
            kernel_size=3,
            bias=True,
            stride=1,
            padding=1,
        )
        self.cca = CCAMask(comp_channels)

    def forward(
        self,
        feature: torch.Tensor,
        cld_comp: torch.Tensor,
        cloudy: torch.Tensor,
    ) -> torch.Tensor:
        z = feature
        for layer in self.net:
            z = layer(z)

        low = self.low_head(z)
        high_z = self.high_refine(z)
        high = self.high_head(high_z - z)
        candidate = low + high
        mask = self.cca(cld_comp)
        return cloudy * (1.0 - mask) + candidate * mask
