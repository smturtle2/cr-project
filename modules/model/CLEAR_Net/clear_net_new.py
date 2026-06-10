from __future__ import annotations

import torch
from torch import nn

from .clear import Extractor, RefineHead, SpectralMaskRouter, Stem


class CLEAR_Net_New(nn.Module):
    def __init__(
        self,
        sar_channels: int = 2,
        cloudy_channels: int = 13,
        out_channels: int = 13,
        feature_layers: int = 2,
        extractor_layers: int = 1,
        candidate_extractor_passes: int = 2,
        num_heads: int = 4,
        return_decomposition: bool = False,
        extractor_dims: tuple[int, ...] | list[int] = (32, 128, 160),
    ):
        super().__init__()
        if out_channels != cloudy_channels:
            raise ValueError("out_channels must match cloudy_channels")
        if candidate_extractor_passes < 0:
            raise ValueError("candidate_extractor_passes must be non-negative")

        extractor_dims = tuple(extractor_dims)
        self.return_decomposition = return_decomposition
        self.extractor_dims = extractor_dims
        self.feature_channels = extractor_dims[0]
        self.fused_channels = self.feature_channels * 2
        self.dim = self.feature_channels
        if self.feature_channels % 2 != 0:
            raise ValueError("extractor_dims[0] must be even for CLD stem concat")
        self.cld_stem_channels = self.feature_channels // 2
        self.candidate_extractor_passes = candidate_extractor_passes

        self.sar_stem = Stem(sar_channels, self.feature_channels)
        self.cld_sar_stem = Stem(sar_channels, self.cld_stem_channels)
        self.cld_hsi_stem = Stem(cloudy_channels, self.cld_stem_channels)

        self.sar_extractor = Extractor(
            self.feature_channels,
            dims=extractor_dims,
            layer_count=extractor_layers,
            num_layers=feature_layers,
            heads=num_heads,
        )
        self.cld_extractor = Extractor(
            self.feature_channels,
            dims=extractor_dims,
            layer_count=extractor_layers,
            num_layers=feature_layers,
            heads=num_heads,
        )

        self.cld_clean_extractor = Extractor(
            self.feature_channels,
            dims=extractor_dims,
            layer_count=extractor_layers,
            num_layers=feature_layers,
            heads=num_heads,
        )
        self.cld_cloudy_extractor = Extractor(
            self.feature_channels,
            dims=extractor_dims,
            layer_count=extractor_layers,
            num_layers=feature_layers,
            heads=num_heads,
        )

        self.fused_refiner = RefineHead(self.fused_channels, self.feature_channels)
        self.candidate_extractor = Extractor(
            self.feature_channels,
            dims=extractor_dims,
            layer_count=candidate_extractor_passes,
            num_layers=feature_layers,
            heads=num_heads,
        )
        self.candidate_head = RefineHead(self.feature_channels, out_channels)
        self.mask_router = SpectralMaskRouter(self.feature_channels, out_channels)

    def forward(self, sar: torch.Tensor, cloudy: torch.Tensor):
        sar_stem = self.sar_stem(sar)
        cld_sar_stem = self.cld_sar_stem(sar)
        cld_hsi_stem = self.cld_hsi_stem(cloudy)
        cld_stem = torch.cat((cld_sar_stem, cld_hsi_stem), dim=1)

        sar_feat = self.sar_extractor(sar_stem)
        cld_feat = self.cld_extractor(cld_stem)

        cld_clean = self.cld_clean_extractor(cld_feat)
        cld_cloudy_raw = cld_feat - cld_clean
        cld_cloudy = self.cld_cloudy_extractor(cld_cloudy_raw)

        fused = torch.cat((sar_feat, cld_clean), dim=1)
        fused = self.fused_refiner(fused)

        candidate_feat = self.candidate_extractor(fused)
        candidate = self.candidate_head(candidate_feat)
        mask_output = self.mask_router(cld_cloudy)
        mask = mask_output["mask"]
        prediction = cloudy * (1.0 - mask) + candidate * mask
        if self.return_decomposition:
            return {
                "prediction": prediction,
                "candidate": candidate,
                "mask": mask,
                "route_weights": mask_output["route_weights"],
                "sar_feat": sar_feat,
                "cld_feat": cld_feat,
                "cld_clean": cld_clean,
                "cld_cloudy": cld_cloudy,
            }
        return prediction
