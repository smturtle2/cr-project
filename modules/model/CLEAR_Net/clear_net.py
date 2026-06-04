from __future__ import annotations

import torch
from torch import nn

from .aca_crnet import ACA_CRNet, init_net
from .clear import Extractor, RefineHead, SpectralMaskRouter, Stem


class CLEAR_Net(nn.Module):
    def __init__(
        self,
        sar_channels: int = 2,
        cloudy_channels: int = 13,
        out_channels: int = 13,
        dim: int = 128,
        feature_layers: int = 2,
        extractor_layers: int = 1,
        cr_layers: int = 16,
        num_heads: int = 4,
        alpha: float = 0.1,
        init_type: str = "kaiming-uniform",
        ca=None,
        ca_kwargs=None,
        return_decomposition: bool = False,
        extractor_dims: tuple[int, ...] | list[int] = (64, 128, 256),
    ):
        super().__init__()
        if out_channels != cloudy_channels:
            raise ValueError("out_channels must match cloudy_channels")

        extractor_dims = tuple(extractor_dims)
        if extractor_dims[0] * 2 != dim:
            raise ValueError("extractor_dims[0] * 2 must match dim")

        self.return_decomposition = return_decomposition
        self.dim = dim
        self.extractor_dims = extractor_dims
        self.fused_extractor_dims = tuple(dim * 2 for dim in extractor_dims)
        self.feature_channels = extractor_dims[0]
        if self.feature_channels % 2 != 0:
            raise ValueError("extractor_dims[0] must be even for CLD stem concat")
        self.cld_stem_channels = self.feature_channels // 2

        # stem 생성
        self.sar_stem = Stem(sar_channels, self.feature_channels)
        self.cld_sar_stem = Stem(sar_channels, self.cld_stem_channels)
        self.cld_hsi_stem = Stem(cloudy_channels, self.cld_stem_channels)

        # feat 생성
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

        # feat 분리
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

        # feat 결합
        self.fused_extractor = Extractor(
            self.dim,
            dims=self.fused_extractor_dims,
            layer_count=extractor_layers,
            num_layers=feature_layers,
            heads=num_heads,
        )
        self.fused_refiner = RefineHead(self.dim, self.dim)

        # 복원 생성
        self.aux_head = RefineHead(self.dim, out_channels)

        decoder_kwargs = {}
        if ca is not None:
            decoder_kwargs["ca"] = ca
        self.aca_crnet = init_net(
            ACA_CRNet(
                out_channels=out_channels,
                alpha=alpha,
                num_layers=cr_layers,
                feature_sizes=self.dim,
                ca_kwargs=ca_kwargs,
                **decoder_kwargs,
            ),
            init_type=init_type,
        )
        self.mask_router = init_net(
            SpectralMaskRouter(self.feature_channels, out_channels),
            init_type=init_type,
        )

    def forward(self, sar: torch.Tensor, cloudy: torch.Tensor):
        # stem 생성
        sar_stem = self.sar_stem(sar)
        cld_sar_stem = self.cld_sar_stem(sar)
        cld_hsi_stem = self.cld_hsi_stem(cloudy)
        cld_stem = torch.cat((cld_sar_stem, cld_hsi_stem), dim=1)

        # feat 생성
        sar_feat = self.sar_extractor(sar_stem)
        cld_feat = self.cld_extractor(cld_stem)

        # feat 분리
        cld_clean = self.cld_clean_extractor(cld_feat)
        cld_cloudy_raw = cld_feat - cld_clean
        cld_cloudy = self.cld_cloudy_extractor(cld_cloudy_raw)

        # feat 결합
        fused = torch.cat((sar_feat, cld_clean), dim=1)
        fused = self.fused_extractor(fused)
        fused = self.fused_refiner(fused)

        # 복원 생성
        aux_clear = self.aux_head(fused)
        candidate = self.aca_crnet(fused)
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
                "aux_clear": aux_clear,
            }
        return prediction
