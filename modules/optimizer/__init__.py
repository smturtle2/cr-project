"""Optimizer 빌더."""

from __future__ import annotations

import torch
from torch import nn


def build_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    betas: tuple[float, float] = (0.9, 0.999),
) -> torch.optim.Optimizer:
    """AdamW optimizer.

    주파수 영역 학습은 gradient scale이 변동성이 크기 때문에
    AdamW의 weight decay로 안정성을 확보한다.
    """
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
    )


__all__ = ["build_optimizer"]
