from __future__ import annotations

import torch
from torch import nn
from torch.nn import init

from .ca_flash import ConAttn
from .clear import Residual3x3Block, SampleUp


class ResBlock(nn.Module):
    def __init__(self, channels: int, alpha: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                bias=False,
                padding=1,
                padding_mode="reflect",
            ),
            nn.ReLU(True),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                bias=False,
                padding=1,
                padding_mode="reflect",
            ),
        )
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * self.net(x)


class AttentionResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        alpha: float = 0.1,
        ca=ConAttn,
        ca_kwargs=None,
    ):
        super().__init__()
        ca_kwargs = {} if ca_kwargs is None else dict(ca_kwargs)
        self.net = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                bias=False,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.ReLU(True),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                bias=False,
                padding=1,
                padding_mode="reflect",
            ),
            nn.ReLU(True),
            ca(
                input_channels=channels,
                output_channels=channels,
                ksize=1,
                stride=1,
                **ca_kwargs,
            ),
        )
        self.up = SampleUp(channels, channels)
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        out = self.up(out)
        return x + self.alpha * out


class CloudMask(nn.Module):
    def __init__(self, cloud_channels: int, mask_channels: int):
        super().__init__()
        self.body = nn.Sequential(
            Residual3x3Block(cloud_channels),
            Residual3x3Block(cloud_channels),
        )
        self.head = nn.Conv2d(cloud_channels, mask_channels, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, cloud_feat: torch.Tensor) -> torch.Tensor:
        return self.activation(self.head(self.body(cloud_feat)))


class ACA_CRNet(nn.Module):
    def __init__(
        self,
        out_channels: int = 13,
        alpha: float = 0.1,
        num_layers: int = 16,
        feature_sizes: int = 256,
        cloud_channels: int | None = None,
        ca=ConAttn,
        ca_kwargs=None,
    ):
        super().__init__()
        ca_kwargs = {} if ca_kwargs is None else dict(ca_kwargs)
        attention_layers = {num_layers // 2, num_layers * 3 // 4}
        self.body = nn.ModuleList(
            [
                AttentionResBlock(
                    feature_sizes,
                    alpha=alpha,
                    ca=ca,
                    ca_kwargs=ca_kwargs,
                )
                if index in attention_layers
                else ResBlock(feature_sizes, alpha=alpha)
                for index in range(num_layers)
            ]
        )
        self.candidate_head = nn.Conv2d(
            feature_sizes,
            out_channels,
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
        )
        self.mask = CloudMask(
            feature_sizes // 2 if cloud_channels is None else cloud_channels,
            out_channels,
        )

    def forward(
        self,
        fused_feature: torch.Tensor,
        cloud_feat: torch.Tensor,
        cloudy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = fused_feature
        for layer in self.body:
            z = layer(z)

        candidate = self.candidate_head(z)
        mask = self.mask(cloud_feat)
        prediction = cloudy * (1.0 - mask) + candidate * mask
        return prediction, candidate, mask


def init_weights(net: nn.Module, init_type: str = "kaiming-uniform", gain: float = 0.02):
    def init_func(module: nn.Module):
        classname = module.__class__.__name__
        if hasattr(module, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(module.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(module.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(module.weight.data, a=0, mode="fan_in")
            elif init_type == "kaiming-uniform":
                init.kaiming_uniform_(module.weight.data, mode="fan_in", nonlinearity="relu")
            elif init_type == "orthogonal":
                init.orthogonal_(module.weight.data, gain=gain)
            else:
                raise NotImplementedError(f"initialization method {init_type} is not implemented")
            if hasattr(module, "bias") and module.bias is not None:
                init.constant_(module.bias.data, 0.0)

    net.apply(init_func)


def init_net(net: nn.Module, init_type: str = "kaiming-uniform") -> nn.Module:
    init_weights(net, init_type=init_type)
    return net
