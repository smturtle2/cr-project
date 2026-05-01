# -*- coding: utf-8 -*-
"""Memory-optimized contextual attention for the current ACA_CRNet path."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConAttn(nn.Module):
    def __init__(
        self,
        input_channels=128,
        output_channels=64,
        ksize=1,
        stride=1,
        rate=1,
        softmax_scale=1.0,
        chunk_size="auto",
    ):
        super().__init__()
        if ksize != 1:
            raise ValueError("ca_optim.ConAttn currently supports ksize=1 only.")
        if stride != 1:
            raise ValueError("ca_optim.ConAttn currently supports stride=1 only.")
        if isinstance(chunk_size, str):
            if chunk_size != "auto":
                raise ValueError("chunk_size must be a positive integer or 'auto'.")
        elif chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")

        self.ksize = ksize
        self.stride = stride
        self.softmax_scale = softmax_scale
        self.chunk_size = chunk_size if chunk_size == "auto" else int(chunk_size)

        self.linear_weight = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels // rate,
                out_channels=input_channels // (4 * rate),
                kernel_size=ksize,
                stride=1,
                padding=ksize // 2,
            ),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(
                input_channels // (4 * rate),
                out_channels=1,
                kernel_size=ksize,
                stride=1,
                padding=ksize // 2,
            ),
        )
        self.bias = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels // rate,
                out_channels=input_channels // (4 * rate),
                kernel_size=ksize,
                stride=1,
                padding=ksize // 2,
            ),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(
                input_channels // (4 * rate),
                out_channels=1,
                kernel_size=ksize,
                stride=1,
                padding=ksize // 2,
            ),
        )
        self.query = nn.Conv2d(
            in_channels=input_channels,
            out_channels=input_channels // rate,
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
            nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x):
        x1 = self.value(x)
        x2 = self.query(x)
        weight = self.linear_weight(x2)
        bias = self.bias(x2)

        y = [
            self._forward_sample(x2_i, x1_i, weight_i, bias_i)
            for x2_i, x1_i, weight_i, bias_i in zip(x2, x1, weight, bias)
        ]
        y = torch.stack(y, dim=0)
        y = self.linear(y)
        return y + x

    def _forward_sample(self, query_map, value_map, weight_map, bias_map):
        query_channels, height, width = query_map.shape
        value_channels = value_map.shape[0]
        num_tokens = height * width
        chunk_size = self._resolve_chunk_size(num_tokens, query_map.dtype, query_map.device)

        query_tokens = query_map.reshape(query_channels, num_tokens)
        key_tokens = query_tokens.transpose(0, 1)
        key_norm = torch.sqrt((key_tokens * key_tokens).sum(dim=1, keepdim=True))
        key_tokens = key_tokens / torch.clamp(key_norm, min=1e-4)

        value_tokens = value_map.reshape(value_channels, num_tokens)
        weight_tokens = weight_map.reshape(num_tokens)
        bias_tokens = bias_map.reshape(num_tokens)
        output = value_tokens.new_empty(value_channels, num_tokens)

        for start in range(0, num_tokens, chunk_size):
            end = min(start + chunk_size, num_tokens)
            query_chunk = query_tokens[:, start:end]
            logits = key_tokens @ query_chunk

            mean = logits.mean(dim=0, keepdim=True)
            sparse = F.relu(logits - mean * weight_tokens[start:end].unsqueeze(0) + bias_tokens[start:end].unsqueeze(0))
            sparse_mask = (sparse != 0.0).to(logits.dtype)

            attn = F.softmax(logits * sparse * self.softmax_scale, dim=0)
            attn = attn * sparse_mask
            attn = attn.clamp(min=1e-8)
            output[:, start:end] = value_tokens @ attn

        return output.reshape(value_channels, height, width)

    def _resolve_chunk_size(self, num_tokens, dtype, device):
        if self.chunk_size != "auto":
            return min(int(self.chunk_size), num_tokens)

        if device.type != "cuda" or not torch.cuda.is_available():
            return min(num_tokens, 4096)

        try:
            free_bytes, _ = torch.cuda.mem_get_info(device)
        except TypeError:
            with torch.cuda.device(device):
                free_bytes, _ = torch.cuda.mem_get_info()
        except RuntimeError:
            return min(num_tokens, 4096)

        element_size = torch.empty((), dtype=dtype, device=device).element_size()
        # logits, sparse, sparse_mask, and attn dominate the chunk-local peak.
        bytes_per_token = max(1, num_tokens * element_size * 4)
        memory_budget = int(free_bytes * 0.25)
        auto_chunk = max(1, memory_budget // bytes_per_token)
        auto_chunk = min(num_tokens, auto_chunk)

        # Keep GEMMs reasonably large while preserving the memory cap above.
        return max(1, auto_chunk)
