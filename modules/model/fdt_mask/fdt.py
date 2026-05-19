from __future__ import annotations

import torch
import torch.nn as nn

from ..fdt.fdt import CommonEncoder, FDT, ResizeConvUp


class ResizeConvUpHalf(ResizeConvUp):
    def __init__(self, in_channels: int, blocks: int = 2):
        super().__init__(in_channels, blocks=blocks)
        self.out_channels = in_channels // 2
        self.refine = nn.Conv2d(
            in_channels,
            self.out_channels,
            kernel_size=1,
        )


class FDTMask(FDT):
    def __init__(
        self,
        sar_channels=2,
        cloudy_channels=13,
        dim=256,
        num_layers=2,
        num_heads=4,
    ):
        super().__init__(
            sar_channels=sar_channels,
            cloudy_channels=cloudy_channels,
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        self.up_dim = dim // 2
        self.common_dim = dim // 2
        self.com_fuse = nn.Conv2d(self.up_dim * 2, self.common_dim, kernel_size=1)
        self.up = ResizeConvUpHalf(dim)

    def forward(self, sar: torch.Tensor, cloudy: torch.Tensor):
        sar_feat_l = self.sar_encoder(sar)
        cld_feat_l = self.cld_encoder(cloudy)

        sar_com_l = self.sar_common_encoder(sar_feat_l)
        cld_com_l = self.cld_common_encoder(cld_feat_l)

        sar_feat = self.up(sar_feat_l)
        cld_feat = self.up(cld_feat_l)
        sar_com = self.up(sar_com_l)
        cld_com = self.up(cld_com_l)
        sar_comp = sar_feat - sar_com
        cld_comp = cld_feat - cld_com

        com_fused = self.com_fuse(torch.cat((sar_com, cld_com), dim=1))
        output = torch.cat((com_fused, sar_comp), dim=1)
        return (
            output,
            sar_com,
            cld_com,
            sar_comp,
            cld_comp,
        )


class MaskEncoder(CommonEncoder):
    def __init__(
        self,
        dim: int,
        out_channels: int,
        num_layers: int,
        heads: int,
    ):
        super().__init__(dim, num_layers, heads)
        self.down = nn.Sequential(
            nn.Conv2d(
                dim // 2,
                dim,
                kernel_size=3,
                stride=2,
                padding=1,
                padding_mode="replicate",
            ),
            nn.GELU(),
        )
        self.up = ResizeConvUpHalf(dim)
        self.out = nn.Conv2d(self.up.out_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.down(x)
        feature = super().forward(feature)
        feature = self.up(feature)
        return torch.sigmoid(self.out(feature))
