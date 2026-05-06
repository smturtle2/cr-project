# -*- coding: utf-8 -*-
"""FlashAttention-compatible contextual attention block."""

from __future__ import annotations

import math
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


class ConAttn(nn.Module):
    def __init__(
        self,
        input_channels=128,
        output_channels=64,
        ksize=1,
        stride=1,
        rate=1,
        softmax_scale=1.0,
        num_heads=4,
        flash_dtype=torch.bfloat16,
        lambda_init=1e-3,
    ):
        super().__init__()
        if ksize != 1:
            raise ValueError("ca_flash.ConAttn currently supports ksize=1 only.")
        if stride != 1:
            raise ValueError("ca_flash.ConAttn currently supports stride=1 only.")
        if input_channels != output_channels:
            raise ValueError(
                "ca_flash.ConAttn requires input_channels == output_channels for residual output."
            )
        if input_channels % rate != 0:
            raise ValueError("input_channels must be divisible by rate.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if flash_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("flash_dtype must be torch.float16 or torch.bfloat16.")

        query_channels = input_channels // rate
        hidden_channels = input_channels // (4 * rate)
        if hidden_channels <= 0:
            raise ValueError("input_channels // (4 * rate) must be positive.")
        if query_channels % num_heads != 0:
            raise ValueError("input_channels // rate must be divisible by num_heads.")
        if input_channels % num_heads != 0:
            raise ValueError("input_channels must be divisible by num_heads.")

        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.softmax_scale = softmax_scale
        self.num_heads = num_heads
        self.flash_dtype = flash_dtype
        self.lambda_scale = nn.Parameter(torch.tensor(float(lambda_init)))

        self.linear_weight = nn.Sequential(
            nn.Conv2d(
                in_channels=query_channels,
                out_channels=hidden_channels,
                kernel_size=ksize,
                stride=1,
                padding=ksize // 2,
            ),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(
                hidden_channels,
                out_channels=1,
                kernel_size=ksize,
                stride=1,
                padding=ksize // 2,
            ),
        )
        self.bias = nn.Sequential(
            nn.Conv2d(
                in_channels=query_channels,
                out_channels=hidden_channels,
                kernel_size=ksize,
                stride=1,
                padding=ksize // 2,
            ),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(
                hidden_channels,
                out_channels=1,
                kernel_size=ksize,
                stride=1,
                padding=ksize // 2,
            ),
        )
        self.query = nn.Conv2d(
            in_channels=input_channels,
            out_channels=query_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.value = nn.Conv2d(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.linear = nn.Sequential(
            nn.Conv2d(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(0.2, True),
        )

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if strict and "lambda_scale" not in state_dict:
            compatible_state_dict = OrderedDict(state_dict)
            if hasattr(state_dict, "_metadata"):
                compatible_state_dict._metadata = state_dict._metadata
            compatible_state_dict["lambda_scale"] = self.lambda_scale.detach().clone()
            state_dict = compatible_state_dict
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _attention_dtype(self, x: torch.Tensor) -> torch.dtype:
        if x.is_cuda:
            return self.flash_dtype
        return x.dtype

    def forward(self, x):
        """Apply output-domain background contrast attention."""
        q_features = self.query(x)
        v_features = self.value(x)

        batch, query_channels, height, width = q_features.shape
        value_channels = v_features.shape[1]
        num_tokens = height * width
        query_head_dim = query_channels // self.num_heads
        value_head_dim = value_channels // self.num_heads

        q_tokens = q_features.flatten(2).transpose(1, 2).contiguous()
        k_tokens = F.normalize(q_tokens.float(), dim=-1, eps=1e-4).to(q_tokens.dtype)
        v_tokens = v_features.flatten(2).transpose(1, 2).contiguous()

        q = (
            q_tokens.view(batch, num_tokens, self.num_heads, query_head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        k = (
            k_tokens.view(batch, num_tokens, self.num_heads, query_head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        v = (
            v_tokens.view(batch, num_tokens, self.num_heads, value_head_dim)
            .transpose(1, 2)
            .contiguous()
        )

        weight = self.linear_weight(q_features).flatten(2).transpose(1, 2).contiguous()
        bias = self.bias(q_features).flatten(2).transpose(1, 2).contiguous()
        weight = weight[:, None, :, :]
        bias = bias[:, None, :, :]

        attn_dtype = self._attention_dtype(q)
        q = (q * (self.softmax_scale * math.sqrt(query_head_dim))).to(attn_dtype)
        k = k.to(attn_dtype)
        v_attn = v.to(attn_dtype)
        weighted_v_attn = (weight.to(attn_dtype) * v_attn).contiguous()

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            y = F.scaled_dot_product_attention(q, k, v_attn, dropout_p=0.0)
            yw = F.scaled_dot_product_attention(q, k, weighted_v_attn, dropout_p=0.0)

        y = y.to(v.dtype)
        yw = yw.to(v.dtype)
        background = yw.mean(dim=2, keepdim=True)
        bias_value = (bias.to(v.dtype) * v).sum(dim=2, keepdim=True)
        contrast = yw - background + bias_value
        gate = F.relu(contrast)
        out = y + F.relu(self.lambda_scale).to(y.dtype) * gate

        out = out.transpose(1, 2).reshape(batch, num_tokens, value_channels)
        out = out.transpose(1, 2).reshape(batch, value_channels, height, width)
        return self.linear(out) + x
