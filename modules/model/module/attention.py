# -*- coding: utf-8 -*-

import numbers

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter


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


class _AmpRMSNorm(nn.Module):
    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]

    def __init__(
        self,
        normalized_shape,
        eps=None,
        elementwise_affine=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(_AmpRMSNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)

    def forward(self, x):
        weight = self.weight
        if weight is not None and weight.dtype != x.dtype:
            weight = weight.to(dtype=x.dtype)
        return F.rms_norm(x, self.normalized_shape, weight, self.eps)

    def extra_repr(self):
        return "{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}".format(**self.__dict__)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super(MultiHeadAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def _reshape_heads(self, x):
        b, n, c = x.shape
        return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _restore_heads(self, x):
        b, _, n, _ = x.shape
        return x.transpose(1, 2).reshape(b, n, self.dim)

    def forward(self, query, tgt=None, value=None):
        is_self_attention = tgt is None and value is None
        if tgt is None:
            tgt = query
        if value is None:
            value = tgt

        q = self._reshape_heads(self.q_proj(query))
        k = self._reshape_heads(self.k_proj(tgt))
        v = self._reshape_heads(self.v_proj(value))

        out = _sdpa(q, k, v)

        if is_self_attention:
            out = out.float()
            v_norm = F.normalize(v.float(), dim=-1, eps=1e-6)
            out = out - (out * v_norm).sum(dim=-1, keepdim=True) * v_norm

        out = self._restore_heads(out.to(query.dtype))
        return self.out_proj(out)


class TransformerLayer(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4):
        super(TransformerLayer, self).__init__()
        self.norm1 = _AmpRMSNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads)
        self.norm2 = _AmpRMSNorm(dim)
        hidden_dim = dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x, tgt=None):
        x = x + self.attn(self.norm1(x), tgt)
        x = x + self.mlp(self.norm2(x))
        return x
