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
        prediction = prediction.float()
        target = target.float()
        return (
            weights[0] * F.l1_loss(prediction, target)
            + weights[1] * (1.0 - loss_fn.ssim(prediction, target))
        )

    def test_combines_prediction_l1_and_ssim_losses(self) -> None:
        loss_fn = FDTCCALoss()
        prediction = torch.rand(2, 13, 16, 16) * 5.0
        candidate = torch.rand(2, 13, 16, 16) * 5.0
        target = torch.rand(2, 13, 16, 16) * 5.0
        model_output = {"prediction": prediction, "candidate": candidate}

        loss = loss_fn(model_output, target)
        expected = self._expected_output_loss(loss_fn, prediction, target)

        self.assertTrue(torch.allclose(loss, expected))

    def test_accepts_prediction_tensor(self) -> None:
        loss_fn = FDTCCALoss()
        prediction = torch.rand(2, 13, 16, 16) * 5.0
        target = torch.rand(2, 13, 16, 16) * 5.0

        loss = loss_fn(prediction, target)
        expected = self._expected_output_loss(loss_fn, prediction, target)

        self.assertTrue(torch.allclose(loss, expected))

    def test_accepts_custom_weights(self) -> None:
        loss_fn = FDTCCALoss(l1_weight=0.7, ssim_weight=0.3)
        prediction = torch.rand(2, 13, 16, 16) * 5.0
        target = torch.rand(2, 13, 16, 16) * 5.0

        loss = loss_fn(prediction, target)
        expected = self._expected_output_loss(
            loss_fn,
            prediction,
            target,
            weights=(0.7, 0.3),
        )

        self.assertTrue(torch.allclose(loss, expected))

    def test_default_ssim_data_range_matches_five_unit_inputs(self) -> None:
        loss_fn = FDTCCALoss()
        unit_range_loss_fn = FDTCCALoss(data_range=1.0)
        prediction = torch.rand(2, 13, 16, 16) * 5.0
        target = torch.rand(2, 13, 16, 16) * 5.0

        ssim = loss_fn.ssim(prediction, target)
        expected = unit_range_loss_fn.ssim(prediction / 5.0, target / 5.0)

        self.assertTrue(torch.allclose(ssim, expected, atol=1e-6))

    def test_factory_returns_tmp_main_compatible_loss_fn(self) -> None:
        loss_fn = make_fdt_cca_loss_fn()
        criterion = FDTCCALoss()
        prediction = torch.rand(2, 13, 16, 16) * 5.0
        target = torch.rand(2, 13, 16, 16) * 5.0

        loss = loss_fn({"prediction": prediction}, {"target": target})
        expected = criterion(prediction, target)

        self.assertTrue(torch.allclose(loss, expected))

    def test_loss_stays_finite_for_large_bfloat16_inputs(self) -> None:
        loss_fn = FDTCCALoss()
        prediction = (torch.rand(2, 13, 16, 16) * 5.0).bfloat16()
        target = (torch.rand(2, 13, 16, 16) * 5.0).bfloat16()

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            loss = loss_fn(prediction, target)

        self.assertTrue(bool(torch.isfinite(loss).item()))
        self.assertGreaterEqual(float(loss), 0.0)

    def test_backward_is_finite(self) -> None:
        loss_fn = FDTCCALoss()
        prediction = (torch.rand(2, 13, 16, 16) * 5.0).requires_grad_()
        candidate = (torch.rand(2, 13, 16, 16) * 5.0).requires_grad_()
        target = torch.rand(2, 13, 16, 16) * 5.0
        model_output = {"prediction": prediction, "candidate": candidate}

        loss = loss_fn(model_output, target)
        loss.backward()

        self.assertTrue(bool(torch.isfinite(loss).item()))
        self.assertIsNotNone(prediction.grad)
        self.assertIsNone(candidate.grad)
        self.assertTrue(bool(torch.isfinite(prediction.grad).all().item()))


if __name__ == "__main__":
    unittest.main()
