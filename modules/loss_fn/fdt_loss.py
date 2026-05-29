from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def _check_same_shape(first: torch.Tensor, second: torch.Tensor) -> None:
    if first.shape != second.shape:
        raise ValueError(f"feature shapes must match: {first.shape} != {second.shape}")


class _CharbonnierLoss(nn.Module):
    def __init__(
        self,
        *,
        eps: float = 1e-3,
    ) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        _check_same_shape(prediction, target)
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            prediction = prediction.float()
            target = target.float()
            return torch.sqrt((prediction - target).square() + self.eps**2).mean()


class _LaplacianLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        self.register_buffer("_kernel", kernel.view(1, 1, 3, 3))

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        _check_same_shape(prediction, target)
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            prediction = prediction.float()
            target = target.float()
            return F.l1_loss(self._laplacian(prediction), self._laplacian(target))

    def _laplacian(self, image: torch.Tensor) -> torch.Tensor:
        channel = image.size(1)
        kernel = self._kernel.to(device=image.device, dtype=image.dtype).expand(
            channel,
            1,
            3,
            3,
        )
        image = F.pad(image, (1, 1, 1, 1), mode="replicate")
        return F.conv2d(image, kernel, groups=channel)


class _SAMLoss(nn.Module):
    def __init__(self, *, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        _check_same_shape(prediction, target)
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            prediction = prediction.float()
            target = target.float()
            cosine = F.cosine_similarity(prediction, target, dim=1, eps=self.eps)
            return torch.acos(cosine.clamp(-1.0 + self.eps, 1.0 - self.eps)).mean()


class FDTCCALoss(nn.Module):
    def __init__(
        self,
        *,
        charbonnier_weight: float = 1.0,
        candidate_loss_weight: float = 0.1,
        laplacian_weight: float = 0.05,
        sam_weight: float = 0.03,
    ) -> None:
        super().__init__()
        self.charbonnier_weight = charbonnier_weight
        self.candidate_loss_weight = candidate_loss_weight
        self.laplacian_weight = laplacian_weight
        self.sam_weight = sam_weight
        self.charbonnier_loss = _CharbonnierLoss()
        self.candidate_loss = _CharbonnierLoss()
        self.laplacian_loss = _LaplacianLoss()
        self.sam_loss = _SAMLoss()

    def forward(
        self,
        model_output: Any,
        target: torch.Tensor,
    ) -> torch.Tensor:
        prediction, candidate = self._split_model_output(model_output)
        _check_same_shape(prediction, target)
        loss = self.charbonnier_weight * self.charbonnier_loss(prediction, target)
        loss = loss + self.laplacian_weight * self.laplacian_loss(prediction, target)
        loss = loss + self.sam_weight * self.sam_loss(prediction, target)
        if self.candidate_loss_weight != 0.0 and candidate is None:
            raise ValueError("FDTCCALoss requires candidate output when candidate_loss_weight is non-zero")
        if candidate is not None:
            _check_same_shape(candidate, target)
            loss = loss + self.candidate_loss_weight * self.candidate_loss(candidate, target)
        return loss

    def _split_model_output(
        self,
        model_output: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(model_output, torch.Tensor):
            return model_output, None
        if not isinstance(model_output, tuple) or len(model_output) < 2:
            raise TypeError("FDTCCALoss expects a tensor or FDT-CCA output tuple")
        prediction = model_output[0]
        candidate = model_output[1]
        if not isinstance(prediction, torch.Tensor):
            raise TypeError("FDT-CCA prediction output must be a tensor")
        if not isinstance(candidate, torch.Tensor):
            raise TypeError("FDT-CCA candidate output must be a tensor")
        return prediction, candidate
