from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..fdt.ACA_CRNet import ACA_CRNet, DefaultConAttn, ResBlock_att
from ..fdt.fdt import FeatureTransformerBase
from ..module.attention import TransformerLayer


class CCA_AttnEncoder(FeatureTransformerBase):
    def __init__(
        self,
        dim: int,
        num_layers: int,
        heads: int,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerLayer(dim, num_heads=heads) for _ in range(num_layers)]
        )

    def forward(self, feature: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        height, width = feature.shape[-2:]
        tokens = self._to_tokens(feature)
        context_tokens = self._to_tokens(context)
        for block in self.blocks:
            tokens = block(tokens, context_tokens)
        return self._to_feature(tokens, height, width)


class CCA_AttnAdapter(nn.Module):
    def __init__(
        self,
        comp_channels: int,
        feature_channels: int,
        num_layers: int = 1,
        heads: int = 4,
    ):
        super().__init__()
        if feature_channels % heads != 0:
            raise ValueError("feature_channels must be divisible by heads")
        self.feature_down = nn.Sequential(
            nn.Conv2d(
                feature_channels,
                feature_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                padding_mode="replicate",
            ),
            nn.GELU(),
            nn.Conv2d(
                feature_channels,
                feature_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                padding_mode="replicate",
            ),
            nn.GELU(),
        )
        self.comp_down = nn.Sequential(
            nn.Conv2d(
                comp_channels,
                feature_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                padding_mode="replicate",
            ),
            nn.GELU(),
            nn.Conv2d(
                feature_channels,
                feature_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                padding_mode="replicate",
            ),
            nn.GELU(),
        )
        self.encoder = CCA_AttnEncoder(feature_channels, num_layers, heads)
        self.refine = nn.Sequential(
            nn.Conv2d(
                feature_channels,
                feature_channels,
                kernel_size=3,
                padding=1,
                groups=feature_channels,
                padding_mode="replicate",
            ),
            nn.GELU(),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=1),
            nn.GELU(),
        )
        self.film_head = nn.Sequential(
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
        nn.init.zeros_(self.film_head[-1].weight)
        nn.init.zeros_(self.film_head[-1].bias)

    def forward(self, feature: torch.Tensor, cld_comp: torch.Tensor) -> torch.Tensor:
        context = self.encoder(
            self.feature_down(feature),
            self.comp_down(cld_comp),
        )
        context = F.interpolate(
            context,
            size=feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        gamma, beta = self.film_head(self.refine(context)).chunk(2, dim=1)
        return feature * (1.0 + gamma) + beta


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
        self.cca_adapter = CCA_AttnAdapter(
            comp_channels,
            feature_sizes,
            num_layers=cca_layers,
            heads=num_heads,
        )
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
        self.cca_adapter.zero_init()

    def forward(self, feature: torch.Tensor, cld_comp: torch.Tensor) -> torch.Tensor:
        out = feature
        for i, layer in enumerate(self.net):
            if i == self.cca_index:
                out = self.cca_adapter(out, cld_comp)
            out = layer(out)
        return out
