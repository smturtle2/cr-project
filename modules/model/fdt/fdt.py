from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..module.attention import TransformerLayer


@dataclass
class FDTDecomposition:
    sar_com: torch.Tensor
    cld_com: torch.Tensor
    sar_comp: torch.Tensor
    cld_comp: torch.Tensor


@dataclass
class FDTOutput:
    feature: torch.Tensor
    lowres: FDTDecomposition
    midres: FDTDecomposition
    highres: FDTDecomposition


class Upsample(nn.Module):
    def __init__(
        self,
        dim: int,
    ):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                dim,
                dim,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GELU(),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.up(feature)

# Enhanced Spatial Attention
class ESA(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        reduced_channels = max(channels // 4, 1)
        self.conv1 = nn.Conv2d(channels, reduced_channels, kernel_size=1)
        self.conv_f = nn.Conv2d(reduced_channels, reduced_channels, kernel_size=1)
        self.conv_max = nn.Conv2d(
            reduced_channels, reduced_channels, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(
            reduced_channels, reduced_channels, kernel_size=3, stride=2
        )
        self.conv3 = nn.Conv2d(
            reduced_channels, reduced_channels, kernel_size=3, padding=1
        )
        self.conv3_ = nn.Conv2d(
            reduced_channels, reduced_channels, kernel_size=3, padding=1
        )
        self.conv4 = nn.Conv2d(reduced_channels, channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        c1_ = self.conv1(feature)
        c1 = self.conv2(c1_)
        pool_kernel = min(7, c1.shape[-2], c1.shape[-1])
        pool_stride = min(3, pool_kernel)
        pooled = F.max_pool2d(c1, kernel_size=pool_kernel, stride=pool_stride)
        context = self.relu(self.conv_max(pooled))
        context = self.relu(self.conv3(context))
        context = self.conv3_(context)
        context = F.interpolate(
            context,
            size=feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        attention = torch.sigmoid(self.conv4(context + self.conv_f(c1_)))
        return feature * attention


class RFDB(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        distilled_channels = channels // 2
        if distilled_channels <= 0:
            raise ValueError("channels must be greater than one")
        self.c1_d = nn.Conv2d(channels, distilled_channels, kernel_size=1)
        self.c1_r = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.c2_d = nn.Conv2d(channels, distilled_channels, kernel_size=1)
        self.c2_r = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.c3_d = nn.Conv2d(channels, distilled_channels, kernel_size=1)
        self.c3_r = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.c4 = nn.Conv2d(channels, distilled_channels, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.c5 = nn.Conv2d(distilled_channels * 4, channels, kernel_size=1)
        self.esa = ESA(channels)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        distilled_c1 = self.act(self.c1_d(feature))
        r_c1 = self.act(self.c1_r(feature) + feature)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = self.act(self.c2_r(r_c1) + r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = self.act(self.c3_r(r_c2) + r_c2)

        r_c4 = self.act(self.c4(r_c3))
        fused = self.c5(
            torch.cat((distilled_c1, distilled_c2, distilled_c3, r_c4), dim=1)
        )
        return self.esa(fused)


class HighResEncoder(nn.Module):
    def __init__(self, in_channels: int, dim: int):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(
                in_channels,
                dim,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="replicate",
            ),
            nn.GELU(),
            RFDB(dim),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.blocks(image)


class CommonGate(nn.Module):
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.channel_key = nn.Conv2d(dim, 1, kernel_size=1)
        self.channel_query = nn.Conv2d(dim, dim, kernel_size=1)
        self.spatial_key = nn.Conv2d(dim, dim, kernel_size=1)
        self.spatial_query = nn.Conv2d(dim, dim, kernel_size=1)
        self.gate_proj = nn.Conv2d(dim * 2, dim, kernel_size=1)

    def forward(
        self,
        target_feat: torch.Tensor,
        other_feat: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, height, width = target_feat.shape

        channel_query = self.channel_query(target_feat).view(
            batch,
            channels,
            height * width,
        )
        channel_key = (
            self.channel_key(other_feat)
            .view(batch, 1, height * width)
            .transpose(1, 2)
            .contiguous()
        )
        channel_relevance = torch.sigmoid(torch.bmm(channel_query, channel_key))
        channel_feature = channel_relevance.unsqueeze(2) * target_feat

        spatial_query = (
            self.spatial_query(target_feat)
            .view(batch, channels, height * width)
            .transpose(1, 2)
            .contiguous()
        )
        spatial_key = (
            self.spatial_key(other_feat)
            .view(batch, channels, height * width)
            .mean(dim=2)
            .unsqueeze(2)
        )
        spatial_relevance = torch.sigmoid(torch.bmm(spatial_query, spatial_key))
        spatial_feature = (
            spatial_relevance.transpose(1, 2)
            .contiguous()
            .view(batch, 1, height, width)
            * target_feat
        )

        return self.gate_proj(torch.cat((channel_feature, spatial_feature), dim=1))


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        dim: int,
        num_layers: int,
        heads: int,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels,
                dim,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="replicate",
            ),
            nn.GELU(),
            nn.Conv2d(
                dim, dim, kernel_size=3, stride=2, padding=1, padding_mode="replicate"
            ),
        )
        self.blocks = nn.ModuleList(
            [TransformerLayer(dim, num_heads=heads) for _ in range(num_layers)]
        )

    @staticmethod
    def _to_tokens(feature: torch.Tensor) -> torch.Tensor:
        return feature.flatten(2).transpose(1, 2).contiguous()

    @staticmethod
    def _to_feature(
        tokens: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        return (
            tokens.transpose(1, 2)
            .reshape(tokens.shape[0], tokens.shape[2], height, width)
            .contiguous()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.proj(x)
        height, width = feature.shape[-2:]
        tokens = self._to_tokens(feature)
        for block in self.blocks:
            tokens = block(tokens)
        return self._to_feature(tokens, height, width)


# Feature Decomposition Transformer
class FDT(nn.Module):
    def __init__(
        self,
        sar_channels=2,
        cloudy_channels=13,
        dim=256,
        num_layers=2,
        num_heads=4,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        if num_layers <= 0:
            raise ValueError("num_layers must be greater than zero")

        self.sar_channels = sar_channels
        self.cloudy_channels = cloudy_channels
        self.dim = dim
        self.num_layers = num_layers
        self.heads = num_heads
        self.num_heads = num_heads
        self.high_dim = dim
        self.common_out_dim = dim // 4
        self.comp_out_dim = (dim - self.common_out_dim) // 2
        if self.common_out_dim + self.comp_out_dim * 2 != dim:
            raise ValueError("dim must support common/complementary projection split")

        self.sar_encoder = Encoder(sar_channels, dim, num_layers, num_heads)
        self.cld_encoder = Encoder(cloudy_channels, dim, num_layers, num_heads)
        self.sar_high_encoder = HighResEncoder(sar_channels, self.high_dim)
        self.cld_high_encoder = HighResEncoder(cloudy_channels, self.high_dim)

        self.sar_low_common_gate = CommonGate(dim, heads=num_heads)
        self.cld_low_common_gate = CommonGate(dim, heads=num_heads)
        self.sar_mid_common_gate = CommonGate(dim, heads=num_heads)
        self.cld_mid_common_gate = CommonGate(dim, heads=num_heads)
        self.sar_high_common_gate = CommonGate(dim, heads=num_heads)
        self.cld_high_common_gate = CommonGate(dim, heads=num_heads)
        self.sar_low_decomp_fuse = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.cld_low_decomp_fuse = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.sar_mid_fuse = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.cld_mid_fuse = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.sar_mid_decomp_fuse = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.cld_mid_decomp_fuse = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.sar_high_fuse = nn.Conv2d(self.high_dim + dim, dim, kernel_size=1)
        self.cld_high_fuse = nn.Conv2d(self.high_dim + dim, dim, kernel_size=1)
        self.com_fuse = nn.Conv2d(dim * 2, self.common_out_dim, kernel_size=1)
        self.sar_comp_proj = nn.Conv2d(dim, self.comp_out_dim, kernel_size=1)
        self.cld_comp_proj = nn.Conv2d(dim, self.comp_out_dim, kernel_size=1)

        self.sar_low_feat_up = Upsample(dim)
        self.cld_low_feat_up = Upsample(dim)
        self.sar_mid_feat_up = Upsample(dim)
        self.cld_mid_feat_up = Upsample(dim)

    @staticmethod
    def _decompose(
        sar_feat: torch.Tensor,
        cld_feat: torch.Tensor,
        sar_gate: torch.Tensor,
        cld_gate: torch.Tensor,
    ) -> FDTDecomposition:
        return FDTDecomposition(
            sar_com=sar_gate * sar_feat,
            cld_com=cld_gate * cld_feat,
            sar_comp=(1.0 - sar_gate) * sar_feat,
            cld_comp=(1.0 - cld_gate) * cld_feat,
        )

    @staticmethod
    def _downsample_input(image: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            image,
            scale_factor=0.5,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

    def _decompose_level(
        self,
        sar_feat: torch.Tensor,
        cld_feat: torch.Tensor,
        sar_gate: CommonGate,
        cld_gate: CommonGate,
    ) -> FDTDecomposition:
        sar_common_gate = torch.sigmoid(sar_gate(sar_feat, cld_feat))
        cld_common_gate = torch.sigmoid(cld_gate(cld_feat, sar_feat))
        return self._decompose(sar_feat, cld_feat, sar_common_gate, cld_common_gate)

    def forward(self, sar: torch.Tensor, cloudy: torch.Tensor) -> FDTOutput:
        sar_low_feat = self.sar_encoder(self._downsample_input(sar))
        cld_low_feat = self.cld_encoder(self._downsample_input(cloudy))
        lowres = self._decompose_level(
            sar_low_feat,
            cld_low_feat,
            self.sar_low_common_gate,
            self.cld_low_common_gate,
        )
        sar_low = self.sar_low_decomp_fuse(
            torch.cat((lowres.sar_com, lowres.sar_comp), dim=1)
        )
        cld_low = self.cld_low_decomp_fuse(
            torch.cat((lowres.cld_com, lowres.cld_comp), dim=1)
        )
        sar_low_up = self.sar_low_feat_up(sar_low)
        cld_low_up = self.cld_low_feat_up(cld_low)

        sar_mid_base = self.sar_encoder(sar)
        cld_mid_base = self.cld_encoder(cloudy)
        sar_mid_feat = self.sar_mid_fuse(torch.cat((sar_mid_base, sar_low_up), dim=1))
        cld_mid_feat = self.cld_mid_fuse(torch.cat((cld_mid_base, cld_low_up), dim=1))
        midres = self._decompose_level(
            sar_mid_feat,
            cld_mid_feat,
            self.sar_mid_common_gate,
            self.cld_mid_common_gate,
        )
        sar_mid = self.sar_mid_decomp_fuse(
            torch.cat((midres.sar_com, midres.sar_comp), dim=1)
        )
        cld_mid = self.cld_mid_decomp_fuse(
            torch.cat((midres.cld_com, midres.cld_comp), dim=1)
        )
        sar_mid_up = self.sar_mid_feat_up(sar_mid)
        cld_mid_up = self.cld_mid_feat_up(cld_mid)

        sar_high = self.sar_high_encoder(sar)
        cld_high = self.cld_high_encoder(cloudy)
        sar_high_feat = self.sar_high_fuse(torch.cat((sar_high, sar_mid_up), dim=1))
        cld_high_feat = self.cld_high_fuse(torch.cat((cld_high, cld_mid_up), dim=1))
        highres = self._decompose_level(
            sar_high_feat,
            cld_high_feat,
            self.sar_high_common_gate,
            self.cld_high_common_gate,
        )

        com_fused = self.com_fuse(
            torch.cat((highres.sar_com, highres.cld_com), dim=1)
        )
        sar_comp = self.sar_comp_proj(highres.sar_comp)
        cld_comp = self.cld_comp_proj(highres.cld_comp)
        feature = torch.cat((com_fused, sar_comp, cld_comp), dim=1)
        return FDTOutput(feature=feature, lowres=lowres, midres=midres, highres=highres)
