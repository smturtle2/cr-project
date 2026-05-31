from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def _sdpa(query, key, value, *, dropout_p=0.0, scale=None):
    kwargs = {"dropout_p": dropout_p}
    if scale is not None:
        kwargs["scale"] = scale
    return F.scaled_dot_product_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        **kwargs,
    )


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
        flash_dtype=None,
        lambda_init=1e-3,
    ):
        super().__init__()
        if ksize != 1:
            raise ValueError("ConAttn supports ksize=1 only")
        if stride != 1:
            raise ValueError("ConAttn supports stride=1 only")
        if input_channels != output_channels:
            raise ValueError("ConAttn requires input_channels == output_channels")
        if input_channels % rate != 0:
            raise ValueError("input_channels must be divisible by rate")
        if input_channels % num_heads != 0:
            raise ValueError("input_channels must be divisible by num_heads")

        query_channels = input_channels // rate
        hidden_channels = input_channels // (4 * rate)
        if query_channels % num_heads != 0:
            raise ValueError("query channels must be divisible by num_heads")
        if hidden_channels <= 0:
            raise ValueError("hidden channels must be positive")

        self.softmax_scale = softmax_scale
        self.num_heads = num_heads
        self.flash_dtype = flash_dtype
        self.lambda_scale = nn.Parameter(torch.tensor(float(lambda_init)))

        self.linear_weight = nn.Sequential(
            nn.Conv2d(
                query_channels,
                hidden_channels,
                kernel_size=ksize,
                padding=ksize // 2,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(
                hidden_channels,
                1,
                kernel_size=ksize,
                padding=ksize // 2,
                padding_mode="reflect",
            ),
        )
        self.bias = nn.Sequential(
            nn.Conv2d(
                query_channels,
                hidden_channels,
                kernel_size=ksize,
                padding=ksize // 2,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(
                hidden_channels,
                1,
                kernel_size=ksize,
                padding=ksize // 2,
                padding_mode="reflect",
            ),
        )
        self.query = nn.Conv2d(input_channels, query_channels, kernel_size=1)
        self.value = nn.Conv2d(input_channels, input_channels, kernel_size=1)
        self.linear = nn.Sequential(
            nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2, True),
        )

    def _attention_dtype(self, x: torch.Tensor) -> torch.dtype:
        if x.is_cuda and self.flash_dtype is not None:
            return self.flash_dtype
        return x.dtype

    def _attention_context(self, x: torch.Tensor):
        if x.is_cuda and x.dtype in (torch.float16, torch.bfloat16):
            return sdpa_kernel(SDPBackend.FLASH_ATTENTION)
        return nullcontext()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_features = self.query(x.contiguous())
        v_features = self.value(x.contiguous())

        batch, query_channels, height, width = q_features.shape
        value_channels = v_features.shape[1]
        num_tokens = height * width
        query_head_dim = query_channels // self.num_heads
        value_head_dim = value_channels // self.num_heads

        q_tokens = q_features.flatten(2).transpose(1, 2).contiguous()
        k_tokens = F.normalize(q_tokens.float(), dim=-1, eps=1e-4).to(q_tokens.dtype)
        v_tokens = v_features.flatten(2).transpose(1, 2).contiguous()

        q = q_tokens.view(
            batch,
            num_tokens,
            self.num_heads,
            query_head_dim,
        ).transpose(1, 2)
        k = k_tokens.view(
            batch,
            num_tokens,
            self.num_heads,
            query_head_dim,
        ).transpose(1, 2)
        v = v_tokens.view(
            batch,
            num_tokens,
            self.num_heads,
            value_head_dim,
        ).transpose(1, 2)

        weight = self.linear_weight(q_features).flatten(2).transpose(1, 2).contiguous()
        bias = self.bias(q_features).flatten(2).transpose(1, 2).contiguous()
        weight = weight[:, None, :, :]
        bias = bias[:, None, :, :]

        attn_dtype = self._attention_dtype(q)
        q = q.to(attn_dtype)
        k = k.to(attn_dtype)
        v_attn = v.to(attn_dtype)
        weighted_v = (weight.to(attn_dtype) * v_attn).contiguous()

        with self._attention_context(q):
            y = _sdpa(q, k, v_attn, scale=self.softmax_scale)
            yw = _sdpa(q, k, weighted_v, scale=self.softmax_scale)

        y = y.to(v.dtype)
        yw = yw.to(v.dtype)
        background = yw.mean(dim=2, keepdim=True)
        bias_value = (bias.to(v.dtype) * v).mean(dim=2, keepdim=True)
        gate = F.relu(yw - background + bias_value)
        out = y + F.relu(self.lambda_scale).to(y.dtype) * gate

        out = out.transpose(1, 2).reshape(batch, num_tokens, value_channels)
        out = out.transpose(1, 2).reshape(batch, value_channels, height, width).contiguous()
        return self.linear(out) + x
