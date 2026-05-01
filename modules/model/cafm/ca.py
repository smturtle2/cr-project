# -*- coding: utf-8 -*-
"""
Attentive Contextual Attention (AC-Attention) Module
원본 ACA-CRNet 코드 — 수정 없이 그대로 사용

@author: Wenli Huang
@paper: "Attentive Contextual Attention for Cloud Removal", IEEE TGRS, 2024
"""

import torch
from torch import nn
from torch.nn import functional as F


class ConAttn(nn.Module):
    def __init__(self, input_channels=128, output_channels=64, ksize=1, stride=1, rate=1, softmax_scale=1.):
        super().__init__()
        self.ksize = ksize
        self.stride = stride
        self.softmax_scale = softmax_scale
        self.linear_weight = nn.Sequential(
            nn.Conv2d(in_channels=input_channels // rate, out_channels=input_channels // (4*rate), kernel_size=ksize, stride=1,
                      padding=ksize // 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(input_channels // (4*rate), out_channels=1, kernel_size=ksize, stride=1,
                      padding=ksize // 2)
        )
        self.bias = nn.Sequential(
            nn.Conv2d(in_channels=input_channels // rate, out_channels=input_channels // (4*rate), kernel_size=ksize, stride=1,
                      padding=ksize // 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(input_channels // (4*rate), out_channels=1, kernel_size=ksize, stride=1,
                      padding=ksize // 2)
        )
        self.query = nn.Conv2d(in_channels=input_channels, out_channels=input_channels//rate, kernel_size=1, stride=1, padding=0)
        self.value = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, stride=1, padding=0)
        self.linear = nn.Sequential(
            nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        x1 = self.value(x)
        x2 = self.query(x)

        x1s = list(x1.size())
        x2s = list(x2.size())

        kernel = self.ksize
        raw_w = extract_patches(x1, kernel=kernel, stride=self.stride)
        raw_w = raw_w.contiguous().view(x1s[0], -1, x1s[1], kernel, kernel)

        raw_w_groups = torch.split(raw_w, 1, dim=0)

        f_groups = torch.split(x2, 1, dim=0)
        weight = self.linear_weight(x2)
        bias = self.bias(x2)
        weight_groups = torch.split(weight, 1, dim=0)
        bias_groups = torch.split(bias, 1, dim=0)
        w = extract_patches(x2, kernel=self.ksize, stride=self.stride)
        w = w.contiguous().view(x2s[0], -1, x2s[1], self.ksize, self.ksize)

        w_groups = torch.split(w, 1, dim=0)
        y = []
        scale = self.softmax_scale
        padding = 0 if self.ksize == 1 else (self.ksize - 1)//2
        for xi, wi, raw_wi, wei, bii in zip(f_groups, w_groups, raw_w_groups, weight_groups, bias_groups):
            wi = wi[0]
            escape_NaN = torch.FloatTensor([1e-4])
            if torch.cuda.is_available():
                escape_NaN = escape_NaN.cuda()
            wi_normed = wi / torch.max(torch.sqrt((wi * wi).sum([1, 2, 3], keepdim=True)), escape_NaN)
            yi = F.conv2d(xi, wi_normed, stride=1, padding=padding)
            yi = yi.contiguous().view(1, x2s[2] // self.stride * x2s[3] // self.stride, x2s[2], x2s[3])

            sparse = F.relu(yi - yi.mean(dim=1, keepdim=True) * wei + bii)
            sparse_r = (sparse != 0.).float()

            yi = yi * sparse
            yi = F.softmax(yi * scale, dim=1)
            yi = yi * sparse_r
            yi = yi.clamp(min=1e-8)

            wi_center = raw_wi[0]
            yi = F.conv_transpose2d(yi, wi_center, stride=1, padding=padding) / (self.ksize * self.ksize)
            y.append(yi)
        y = torch.cat(y, dim=0)

        y.contiguous().view(x1s)
        y = self.linear(y)
        y = y + x
        return y


def extract_patches(x, kernel=3, stride=1):
    if kernel != 1:
        x = nn.ZeroPad2d((kernel-1) //2)(x)
    x = x.permute(0, 2, 3, 1)
    all_patches = x.unfold(1, kernel, stride).unfold(2, kernel, stride)
    return all_patches
