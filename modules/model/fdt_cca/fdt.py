from __future__ import annotations

import torch
import torch.nn as nn

from ..fdt.fdt import FDT, ResizeConvUp
from ..fdt.fdt import CommonEncoder as FeatureEncoder


class JointEncoder(FeatureEncoder):
    def __init__(
        self,
        dim: int,
        num_layers: int,
        heads: int,
    ):
        super().__init__(dim, num_layers, heads)
        self.proj = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(
                dim,
                dim,
                kernel_size=3,
                padding=1,
                padding_mode="replicate",
            ),
        )

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return super().forward(self.proj(torch.cat((first, second), dim=1)))


class ResizeConvUpHalf(ResizeConvUp):
    def __init__(self, in_channels: int, blocks: int = 2):
        super().__init__(in_channels, blocks=blocks)
        self.out_channels = in_channels // 2
        self.refine = nn.Conv2d(
            in_channels,
            self.out_channels,
            kernel_size=1,
        )


class FDT_CCA(FDT):
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
        self.joint_encoder = JointEncoder(dim, num_layers, num_heads)
        self.sar_feat_encoder = FeatureEncoder(dim, num_layers, num_heads)
        self.cld_feat_encoder = FeatureEncoder(dim, num_layers, num_heads)

    def forward(self, sar: torch.Tensor, cloudy: torch.Tensor):
        sar_base_l = self.sar_encoder(sar)
        cld_base_l = self.cld_encoder(cloudy)
        joint_l = self.joint_encoder(sar_base_l, cld_base_l)

        sar_feat_l = self.sar_feat_encoder(sar_base_l + joint_l)
        cld_feat_l = self.cld_feat_encoder(cld_base_l + joint_l)

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
