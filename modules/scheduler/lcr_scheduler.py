from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import LRScheduler


class _LCRWarmupCosineLR(LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        total_epochs: int,
        warmup_epochs: int,
        max_lr: float,
        min_lr: float,
    ) -> None:
        if total_epochs <= warmup_epochs:
            raise ValueError("total_epochs must be greater than warmup_epochs")
        if warmup_epochs <= 0:
            raise ValueError("warmup_epochs must be greater than zero")
        if min_lr <= 0.0:
            raise ValueError("min_lr must be greater than zero")
        if min_lr >= max_lr:
            raise ValueError("min_lr must be smaller than max_lr")

        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.max_lr = max_lr
        self.min_lr = min_lr

        for param_group in optimizer.param_groups:
            param_group["lr"] = max_lr

        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        if self.last_epoch <= self.warmup_epochs:
            return [self._warmup_lr() for _ in self.base_lrs]
        if self.last_epoch >= self.total_epochs:
            return [self.min_lr for _ in self.base_lrs]
        return [self._cosine_lr() for _ in self.base_lrs]

    def _warmup_lr(self) -> float:
        progress = self.last_epoch / self.warmup_epochs
        return self.min_lr + (self.max_lr - self.min_lr) * progress

    def _cosine_lr(self) -> float:
        cosine_progress = (self.last_epoch - self.warmup_epochs) / (
            self.total_epochs - self.warmup_epochs
        )
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
            1.0 + math.cos(math.pi * cosine_progress)
        )


def build_lcr_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_epochs: int,
    warmup_epochs: int = 5,
    max_lr: float = 2e-4,
    min_lr: float = 1e-5,
) -> LRScheduler:
    return _LCRWarmupCosineLR(
        optimizer,
        total_epochs=total_epochs,
        warmup_epochs=warmup_epochs,
        max_lr=max_lr,
        min_lr=min_lr,
    )
