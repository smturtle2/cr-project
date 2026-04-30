# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=4, use_xsa=True):
        super(MultiHeadAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_xsa = use_xsa
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

    def forward(self, query, context=None):
        is_self_attention = context is None
        if context is None:
            context = query

        q = self._reshape_heads(self.q_proj(query))
        k = self._reshape_heads(self.k_proj(context))
        v = self._reshape_heads(self.v_proj(context))

        q_attn = q.to(torch.float16)
        k_attn = k.to(torch.float16)
        v_attn = v.to(torch.float16)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(q_attn, k_attn, v_attn, dropout_p=0.0)

        if self.use_xsa and is_self_attention:
            v_norm = F.normalize(v_attn, dim=-1)
            out = out - (out * v_norm).sum(dim=-1, keepdim=True) * v_norm

        out = self._restore_heads(out.to(query.dtype))
        return self.out_proj(out)


class TransformerLayer(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4, use_xsa=True):
        super(TransformerLayer, self).__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, use_xsa=use_xsa)
        self.norm2 = nn.RMSNorm(dim)
        hidden_dim = dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x, context=None):
        x = x + self.attn(self.norm1(x), context)
        x = x + self.mlp(self.norm2(x))
        return x
