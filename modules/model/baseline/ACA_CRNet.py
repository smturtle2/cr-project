# -*- coding: utf-8 -*-
"""
Created on May 9 10:24:49 2024

@author: Wenli Huang
"""

import torch.nn as nn
from collections import OrderedDict
from torch.nn import init
import torch
from .ca import ConAttn
from ..gate import build_gate_estimator
from ..module.base_module import BaseModule

DefaultConAttn = ConAttn

#resnet block
class ResBlock(nn.Module):
    def __init__(self,in_channels,out_channels=256,alpha=0.1):
        super(ResBlock, self).__init__()
        m = OrderedDict()
        m['conv1']=nn.Conv2d(in_channels, out_channels,kernel_size=3,bias=False,stride=1,padding=1)
        m['relu1']=nn.ReLU(True)
        m['conv2']=nn.Conv2d(out_channels,out_channels, kernel_size=3,bias=False,stride=1,padding=1)
        self.net = nn.Sequential(m)
        self.relu= nn.Sequential(nn.ReLU(True))
        self.alpha = alpha
    #out = 256+2*0-1*(3-1)-1 +1 = 256
    def forward(self,x):
        out =self.net(x)
        out = self.alpha*out + x
        return out

#resnet模块
class ResBlock_att(nn.Module):
    def __init__(self,in_channels,out_channels=256,alpha=0.1,ca=DefaultConAttn,ca_kwargs=None):
        super(ResBlock_att, self).__init__()
        ca_kwargs = {} if ca_kwargs is None else dict(ca_kwargs)
        m = OrderedDict()
        m['conv1']=nn.Conv2d(in_channels, out_channels,kernel_size=3,bias=False,stride=2,padding=1)
        m['relu1']=nn.ReLU(True)
        m['conv2']=nn.Conv2d(out_channels,out_channels, kernel_size=3,bias=False,stride=1,padding=1)
        m['relu2'] = nn.ReLU(True)
        m['att'] = ca(input_channels=out_channels, output_channels=out_channels, ksize=1, stride=1, **ca_kwargs)
        self.net = nn.Sequential(m)
        #self.relu= nn.Sequential(nn.ReLU(True))
        self.alpha = alpha
    #out = 256+2*0-1*(3-1)-1 +1 = 256
    def forward(self,x):
        out =self.net(x)
        out = torch.nn.functional.interpolate(out, scale_factor=2, mode='bilinear',
                                            align_corners=True)
        out = self.alpha*out + x
        return out

def init_net(net, init_type="kaiming-uniform", gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net

# see section 6.2.1
def init_weights(net, init_type="kaiming-uniform", gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == "kaiming-uniform":
                init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class ACA_CRNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        alpha=0.1,
        num_layers=16,
        feature_sizes=256,
        gpu_ids=[],
        ca=DefaultConAttn,
        ca_kwargs=None,
        is_baseline=False,
        gate_mode="mask",
        gate_feat_dim=32,
        gate_prior_weight=0.5,
    ):
        super(ACA_CRNet,self).__init__()
        ca_kwargs = {} if ca_kwargs is None else dict(ca_kwargs)
        sar_channels = in_channels - out_channels
        self.gate_mode = gate_mode
        self.last_gate = None
        self.last_gates = []
        self.gate_estimator = build_gate_estimator(
            gate_mode,
            sar_channels=sar_channels,
            optical_channels=out_channels,
            feat_dim=gate_feat_dim,
            prior_weight=gate_prior_weight,
        )
        m= []
        m.append(nn.Conv2d(in_channels,out_channels=feature_sizes,kernel_size=3,bias=True,stride = 1 ,padding=1))
        m.append(nn.ReLU(True))
        for i in range(num_layers):

            #if i == 6:
            #    m.append(ConAttn(input_channels=feature_sizes, output_channels=feature_sizes, ksize=1, stride=1))

            if i == num_layers//2:
                m.append(ResBlock_att(feature_sizes,feature_sizes,alpha,ca=ca,ca_kwargs=ca_kwargs))# 256/s==int
                if not is_baseline:
                    m.append(BaseModule(sar_channels, out_channels, feature_sizes))
            elif i == num_layers*3//4:
                m.append(ResBlock_att(feature_sizes,feature_sizes,alpha,ca=ca,ca_kwargs=ca_kwargs))# 256/s==int
                if not is_baseline:
                    m.append(BaseModule(sar_channels, out_channels, feature_sizes))
            else:
                m.append(ResBlock(feature_sizes, feature_sizes, alpha))

            #if i == 10:
            #    m.append(ConAttn(input_channels=feature_sizes, output_channels=feature_sizes, ksize=1, stride=1))
        m.append(nn.Conv2d(feature_sizes,out_channels,kernel_size=3, bias=True,stride=1,padding=1))
        self.net = nn.ModuleList(m)
        self.gpu_ids=gpu_ids
        if len(self.gpu_ids) > 0:
            assert(torch.cuda.is_available())
            self.net.to(self.gpu_ids[0])
        if is_baseline:
            init_weights(self.net,"kaiming-uniform")
    
    def forward(self, sar, cloudy):
        x = torch.cat((sar, cloudy), dim=1)
        out = x
        if self.gate_estimator is None:
            gate = None
            self.last_gate = None
        else:
            gate = self.gate_estimator(sar, cloudy)
            self.last_gate = gate
        self.last_gates = []
        for layer in self.net:
            if isinstance(layer, BaseModule):
                out = layer(sar, cloudy, out, gate=gate)
                self.last_gates.append(layer.last_gate)
            else:
                out = layer(out)
        if gate is None and self.last_gates:
            self.last_gate = self.last_gates[-1]
        return cloudy + out
    
