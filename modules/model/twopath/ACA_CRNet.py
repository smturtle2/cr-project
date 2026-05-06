import torch

from ..baseline.ACA_CRNet import ACA_CRNet
from ..module.base_module import BaseModule


class ACA_CRNetRaw(ACA_CRNet):
    def forward(self, sar, cloudy):
        x = torch.cat((sar, cloudy), dim=1)
        out = x
        for layer in self.net:
            if isinstance(layer, BaseModule):
                out = layer(sar, cloudy, out)
            else:
                out = layer(out)
        return out
