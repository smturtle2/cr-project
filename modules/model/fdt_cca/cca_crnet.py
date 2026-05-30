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
        mask_bias_init: float = -5.0,
    ):
        super().__init__()
        self.mask_bias_init = mask_bias_init
        self.body = nn.Sequential(
            Residual3x3Block(cloud_channels),
            Residual3x3Block(cloud_channels),
        )
        self.mask_head = nn.Conv2d(cloud_channels, mask_channels, kernel_size=1)
        self.activation = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.mask_head.weight)
        nn.init.constant_(self.mask_head.bias, self.mask_bias_init)

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
        delta_head = self.net[-1]
        self.net = nn.ModuleList(self.net[:-1])
        self.delta_head = delta_head
        self.cca = CCAMask(cloud_channels, out_channels)

    def forward(
        self,
        feature: torch.Tensor,
        cld_cloud: torch.Tensor,
        cloudy: torch.Tensor,
        *,
        return_candidate: bool = False,
        return_mask: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        z = feature
        for layer in self.net:
            z = layer(z)

        delta = self.delta_head(z)
        mask = self.cca(cld_cloud)
        prediction = cloudy + mask * delta
        candidate = cloudy + delta
        if return_candidate and return_mask:
            return prediction, candidate, mask
        if return_candidate:
            return prediction, candidate
        if return_mask:
            return prediction, mask
        return prediction
