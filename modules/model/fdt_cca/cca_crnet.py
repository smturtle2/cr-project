from __future__ import annotations

import torch
import torch.nn as nn

from ..fdt.ACA_CRNet import ACA_CRNet, DefaultConAttn, ResBlock_att


class CCAFiLM(nn.Module):
    def __init__(self, comp_channels: int, feature_channels: int):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(
                comp_channels,
                feature_channels,
                kernel_size=3,
                padding=1,
                padding_mode="replicate",
            ),
            nn.GELU(),
            nn.Conv2d(
                feature_channels,
                feature_channels,
                kernel_size=3,
                padding=1,
                padding_mode="replicate",
            ),
            nn.GELU(),
            nn.Conv2d(feature_channels, feature_channels * 2, kernel_size=1),
        )
        self.zero_init()

    def zero_init(self) -> None:
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, feature: torch.Tensor, cld_comp: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.adapter(cld_comp).chunk(2, dim=1)
        return feature * (1.0 + gamma) + beta


class CCA_CRNet(ACA_CRNet):
    def __init__(
        self,
        out_channels: int,
        alpha: float = 0.1,
        num_layers: int = 16,
        feature_sizes: int = 256,
        comp_channels: int | None = None,
        ca=DefaultConAttn,
        ca_kwargs=None,
    ):
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
        self.cca_film = CCAFiLM(comp_channels, feature_sizes)
        self.cca_index = self._find_first_attn()

    def _find_first_attn(self) -> int | None:
        return next(
            (
                i
                for i, layer in enumerate(self.net)
                if isinstance(layer, ResBlock_att)
            ),
            None,
        )

    def zero_init_cca(self) -> None:
        self.cca_film.zero_init()

    def forward(self, feature: torch.Tensor, cld_comp: torch.Tensor) -> torch.Tensor:
        out = feature
        for i, layer in enumerate(self.net):
            if i == self.cca_index:
                out = self.cca_film(out, cld_comp)
            out = layer(out)
        return out
