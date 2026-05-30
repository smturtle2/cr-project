from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from modules.loss_fn import (
    FDTCCALoss,
    make_fdt_cca_loss_fn,
)


class FDTCCALossTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def _expected_output_loss(
        self,
        loss_fn: FDTCCALoss,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        weights: tuple[float, float] = (0.9, 0.1),
    ) -> torch.Tensor:
        prediction = prediction.float() / loss_fn.input_scale
        target = target.float() / loss_fn.input_scale
        return (
            weights[0] * F.l1_loss(prediction, target)
            + weights[1] * (1.0 - loss_fn.ssim(prediction, target))
        )

    def test_combines_prediction_l1_and_ssim_losses(self) -> None:
        loss_fn = FDTCCALoss()
        prediction = torch.rand(2, 13, 16, 16) * loss_fn.input_scale
        candidate = torch.rand(2, 13, 16, 16) * loss_fn.input_scale
        target = torch.rand(2, 13, 16, 16) * loss_fn.input_scale
        model_output = (prediction, candidate, None, None, None, None)

        loss = loss_fn(model_output, target)
        expected = self._expected_output_loss(loss_fn, prediction, target)

        self.assertTrue(torch.allclose(loss, expected))

    def test_accepts_prediction_tensor(self) -> None:
        loss_fn = FDTCCALoss()
        prediction = torch.rand(2, 13, 16, 16) * loss_fn.input_scale
        target = torch.rand(2, 13, 16, 16) * loss_fn.input_scale

        loss = loss_fn(prediction, target)
        expected = self._expected_output_loss(loss_fn, prediction, target)

        self.assertTrue(torch.allclose(loss, expected))

    def test_accepts_custom_weights(self) -> None:
        loss_fn = FDTCCALoss(l1_weight=0.7, ssim_weight=0.3)
        prediction = torch.rand(2, 13, 16, 16) * loss_fn.input_scale
        target = torch.rand(2, 13, 16, 16) * loss_fn.input_scale

        loss = loss_fn(prediction, target)
        expected = self._expected_output_loss(
            loss_fn,
            prediction,
            target,
            weights=(0.7, 0.3),
        )

        self.assertTrue(torch.allclose(loss, expected))

    def test_factory_returns_tmp_main_compatible_loss_fn(self) -> None:
        loss_fn = make_fdt_cca_loss_fn()
        criterion = FDTCCALoss()
        prediction = torch.rand(2, 13, 16, 16) * criterion.input_scale
        target = torch.rand(2, 13, 16, 16) * criterion.input_scale

        loss = loss_fn((prediction, None, None), {"target": target})
        expected = criterion(prediction, target)

        self.assertTrue(torch.allclose(loss, expected))

    def test_loss_stays_finite_for_large_bfloat16_inputs(self) -> None:
        loss_fn = FDTCCALoss()
        prediction = (torch.rand(2, 13, 16, 16) * loss_fn.input_scale).bfloat16()
        target = (torch.rand(2, 13, 16, 16) * loss_fn.input_scale).bfloat16()

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            loss = loss_fn(prediction, target)

        self.assertTrue(bool(torch.isfinite(loss).item()))
        self.assertGreaterEqual(float(loss), 0.0)

    def test_backward_is_finite(self) -> None:
        loss_fn = FDTCCALoss()
        prediction = (torch.rand(2, 13, 16, 16) * loss_fn.input_scale).requires_grad_()
        candidate = (torch.rand(2, 13, 16, 16) * loss_fn.input_scale).requires_grad_()
        target = torch.rand(2, 13, 16, 16) * loss_fn.input_scale
        model_output = (prediction, candidate, None, None, None, None)

        loss = loss_fn(model_output, target)
        loss.backward()

        self.assertTrue(bool(torch.isfinite(loss).item()))
        self.assertIsNotNone(prediction.grad)
        self.assertIsNone(candidate.grad)
        self.assertTrue(bool(torch.isfinite(prediction.grad).all().item()))


if __name__ == "__main__":
    unittest.main()
