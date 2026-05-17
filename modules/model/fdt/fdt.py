from __future__ import annotations

from dataclasses import dataclass
import math

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


class DecompositionBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.norm_eps = 1e-6
        self.max_logit_scale = math.log(6.0)
        self.channel_key = nn.Conv2d(dim, 1, kernel_size=1)
        self.channel_query = nn.Conv2d(dim, dim, kernel_size=1)
        self.spatial_key = nn.Conv2d(dim, dim, kernel_size=1)
        self.spatial_query = nn.Conv2d(dim, dim, kernel_size=1)
        self.channel_logit_scale = nn.Parameter(torch.tensor(math.log(2.0)))
        self.spatial_logit_scale = nn.Parameter(torch.tensor(math.log(3.0)))
        self.common_fuse = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.comp_fuse = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.last_relevance_stats: dict[str, dict[str, float]] | None = None

    def _bounded_scale(self, logit_scale: torch.Tensor) -> torch.Tensor:
        return logit_scale.clamp(max=self.max_logit_scale).exp()

    @staticmethod
    def _relevance_stats(
        relevance: torch.Tensor,
        *,
        scale: torch.Tensor,
    ) -> dict[str, float]:
        relevance = relevance.detach().float()
        return {
            "mean": float(relevance.mean().item()),
            "std": float(relevance.std(unbiased=False).item()),
            "min": float(relevance.min().item()),
            "max": float(relevance.max().item()),
            "low_frac": float((relevance < 0.05).float().mean().item()),
            "high_frac": float((relevance > 0.95).float().mean().item()),
            "scale": float(scale.detach().float().item()),
        }

    def _record_relevance_stats(
        self,
        *,
        channel_relevance: torch.Tensor,
        spatial_relevance: torch.Tensor,
    ) -> None:
        self.last_relevance_stats = {
            "channel": self._relevance_stats(
                channel_relevance,
                scale=self._bounded_scale(self.channel_logit_scale),
            ),
            "spatial": self._relevance_stats(
                spatial_relevance,
                scale=self._bounded_scale(self.spatial_logit_scale),
            ),
        }

    def _channel_relevance(
        self,
        target_feat: torch.Tensor,
        other_feat: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, height, width = target_feat.shape
        num_tokens = height * width

        channel_query = self.channel_query(target_feat).view(
            batch,
            channels,
            num_tokens,
        )
        channel_key = self.channel_key(other_feat).view(batch, 1, num_tokens)
        channel_query = F.normalize(channel_query, dim=2, eps=self.norm_eps)
        channel_key = F.normalize(channel_key, dim=2, eps=self.norm_eps)

        scale = self._bounded_scale(self.channel_logit_scale).to(
            dtype=channel_query.dtype
        )
        channel_logits = torch.bmm(
            channel_query,
            channel_key.transpose(1, 2),
        ) * scale
        return torch.sigmoid(channel_logits)

    def _spatial_relevance(
        self,
        target_feat: torch.Tensor,
        other_feat: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, height, width = target_feat.shape
        num_tokens = height * width

        spatial_query = (
            self.spatial_query(target_feat)
            .view(batch, channels, num_tokens)
            .transpose(1, 2)
            .contiguous()
        )
        spatial_key = (
            self.spatial_key(other_feat)
            .view(batch, channels, num_tokens)
            .mean(dim=2)
        )
        spatial_query = F.normalize(spatial_query, dim=2, eps=self.norm_eps)
        spatial_key = F.normalize(spatial_key, dim=1, eps=self.norm_eps)

        scale = self._bounded_scale(self.spatial_logit_scale).to(
            dtype=spatial_query.dtype
        )
        spatial_logits = torch.bmm(spatial_query, spatial_key.unsqueeze(2)) * scale
        return torch.sigmoid(spatial_logits)

    def forward(
        self,
        target_feat: torch.Tensor,
        other_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, height, width = target_feat.shape

        channel_relevance = self._channel_relevance(target_feat, other_feat)
        channel_mask = channel_relevance.unsqueeze(2)
        channel_common = channel_mask * target_feat
        channel_comp = (1.0 - channel_mask) * target_feat

        spatial_relevance = self._spatial_relevance(target_feat, other_feat)
        spatial_mask = (
            spatial_relevance.transpose(1, 2)
            .contiguous()
            .view(batch, 1, height, width)
        )
        spatial_common = spatial_mask * target_feat
        spatial_comp = (1.0 - spatial_mask) * target_feat
        self._record_relevance_stats(
            channel_relevance=channel_relevance,
            spatial_relevance=spatial_relevance,
        )

        common = self.common_fuse(torch.cat((spatial_common, channel_common), dim=1))
        comp = self.comp_fuse(torch.cat((spatial_comp, channel_comp), dim=1))
        return common, comp


class BidirectionalDecompositionBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.sar = DecompositionBlock(dim)
        self.cloudy = DecompositionBlock(dim)

    def forward(
        self,
        sar_feat: torch.Tensor,
        cld_feat: torch.Tensor,
    ) -> FDTDecomposition:
        sar_com, sar_comp = self.sar(sar_feat, cld_feat)
        cld_com, cld_comp = self.cloudy(cld_feat, sar_feat)
        return FDTDecomposition(
            sar_com=sar_com,
            cld_com=cld_com,
            sar_comp=sar_comp,
            cld_comp=cld_comp,
        )


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

        self.low_decomp = BidirectionalDecompositionBlock(dim)
        self.mid_decomp = BidirectionalDecompositionBlock(dim)
        self.high_decomp = BidirectionalDecompositionBlock(dim)
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
    def _downsample_input(image: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            image,
            scale_factor=0.5,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

    def forward(self, sar: torch.Tensor, cloudy: torch.Tensor) -> FDTOutput:
        sar_low_feat = self.sar_encoder(self._downsample_input(sar))
        cld_low_feat = self.cld_encoder(self._downsample_input(cloudy))
        lowres = self.low_decomp(sar_low_feat, cld_low_feat)
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
        midres = self.mid_decomp(sar_mid_feat, cld_mid_feat)
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
        highres = self.high_decomp(sar_high_feat, cld_high_feat)

        com_fused = self.com_fuse(
            torch.cat((highres.sar_com, highres.cld_com), dim=1)
        )
        sar_comp = self.sar_comp_proj(highres.sar_comp)
        cld_comp = self.cld_comp_proj(highres.cld_comp)
        feature = torch.cat((com_fused, sar_comp, cld_comp), dim=1)
        return FDTOutput(feature=feature, lowres=lowres, midres=midres, highres=highres)
