from __future__ import annotations

import torch
from torch import nn

from .aca_crnet import ACA_CRNet, init_net
from .clear import Extractor, RefineHead, Stem


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
        self.fused_extractor_dims = tuple(channel * 2 for channel in extractor_dims)
        self.feature_channels = extractor_dims[0]

        self.sar_stem = Stem(sar_channels, self.feature_channels)
        self.cloudy_stem = Stem(cloudy_channels + sar_channels, self.feature_channels)
        self.sar_extractor = Extractor(
            self.feature_channels,
            dims=extractor_dims,
            layer_count=extractor_layers,
            num_layers=feature_layers,
            heads=num_heads,
        )
        self.cloudy_extractor = Extractor(
            self.feature_channels,
            dims=extractor_dims,
            layer_count=extractor_layers,
            num_layers=feature_layers,
            heads=num_heads,
        )
        self.clear_extractor = Extractor(
            self.feature_channels,
            dims=extractor_dims,
            layer_count=extractor_layers,
            num_layers=feature_layers,
            heads=num_heads,
        )
        self.fused_extractor = Extractor(
            self.dim,
            dims=self.fused_extractor_dims,
            layer_count=extractor_layers,
            num_layers=feature_layers,
            heads=num_heads,
        )
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
                cloud_channels=self.feature_channels,
                ca_kwargs=ca_kwargs,
                **decoder_kwargs,
            ),
            init_type=init_type,
        )

    def forward(self, sar: torch.Tensor, cloudy: torch.Tensor):
        sar_feat = self.sar_extractor(self.sar_stem(sar))
        cloudy_feat = self.cloudy_extractor(
            self.cloudy_stem(torch.cat((sar, cloudy), dim=1))
        )
        clear_feat = self.clear_extractor(cloudy_feat)
        cloud_feat = cloudy_feat - clear_feat
        fused = torch.cat((sar_feat, clear_feat), dim=1)
        fused = self.fused_extractor(fused)

        aux_clear = self.aux_head(fused)
        cr_output = self.aca_crnet(fused, cloud_feat, cloudy)
        prediction = cr_output["prediction"]
        if self.return_decomposition:
            return {
                "prediction": prediction,
                "candidate": cr_output["candidate"],
                "mask": cr_output["mask"],
                "route_weights": cr_output["route_weights"],
                "sar_feat": sar_feat,
                "clear_feat": clear_feat,
                "cloud_feat": cloud_feat,
                "aux_clear": aux_clear,
            }
        return prediction
