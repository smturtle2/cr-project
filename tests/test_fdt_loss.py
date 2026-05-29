from __future__ import annotations

import unittest

import torch

from modules.loss_fn import (
    FDTCCALoss,
)


class FDTCCALossTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_combines_prediction_candidate_log_sam_and_ssim_losses(self) -> None:
        loss_fn = FDTCCALoss(
            charbonnier_weight=1.0,
            candidate_loss_weight=0.1,
            log_weight=0.05,
            sam_weight=0.02,
            ssim_weight=0.1,
        )
        prediction = torch.randn(2, 13, 16, 16)
        candidate = torch.randn(2, 13, 16, 16)
        target = torch.randn(2, 13, 16, 16)
        model_output = (prediction, candidate, None, None, None, None)

        loss = loss_fn(model_output, target)
        expected = (
            loss_fn.charbonnier_loss(prediction, target)
            + 0.1 * loss_fn.candidate_loss(candidate, target)
            + 0.05 * loss_fn.log_loss(prediction, target)
            + 0.02 * loss_fn.sam_loss(prediction, target)
            + 0.1 * loss_fn.ssim_loss(prediction, target)
        )

        self.assertTrue(torch.allclose(loss, expected))

    def test_accepts_prediction_tensor_when_candidate_loss_is_disabled(self) -> None:
        loss_fn = FDTCCALoss(candidate_loss_weight=0.0)
        prediction = torch.randn(2, 13, 16, 16)
        target = torch.randn(2, 13, 16, 16)

        loss = loss_fn(prediction, target)
        expected = (
            loss_fn.charbonnier_loss(prediction, target)
            + 0.05 * loss_fn.log_loss(prediction, target)
            + 0.02 * loss_fn.sam_loss(prediction, target)
            + 0.1 * loss_fn.ssim_loss(prediction, target)
        )

        self.assertTrue(torch.allclose(loss, expected))

    def test_accepts_laplacian_weight_as_log_weight_alias(self) -> None:
        loss_fn = FDTCCALoss(laplacian_weight=0.2)

        self.assertEqual(loss_fn.log_weight, 0.2)

    def test_loss_stays_finite_for_large_bfloat16_inputs(self) -> None:
        loss_fn = FDTCCALoss(candidate_loss_weight=0.0)
        prediction = (torch.randn(2, 13, 16, 16) * 1000.0).bfloat16()
        target = (torch.randn(2, 13, 16, 16) * 1000.0).bfloat16()

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            loss = loss_fn(prediction, target)

        self.assertTrue(bool(torch.isfinite(loss).item()))
        self.assertGreaterEqual(float(loss), 0.0)

    def test_backward_is_finite(self) -> None:
        loss_fn = FDTCCALoss()
        prediction = torch.randn(2, 13, 16, 16, requires_grad=True)
        candidate = torch.randn(2, 13, 16, 16, requires_grad=True)
        target = torch.randn(2, 13, 16, 16)
        model_output = (prediction, candidate, None, None, None, None)

        loss = loss_fn(model_output, target)
        loss.backward()

        self.assertTrue(bool(torch.isfinite(loss).item()))
        self.assertIsNotNone(prediction.grad)
        self.assertIsNotNone(candidate.grad)
        self.assertTrue(bool(torch.isfinite(prediction.grad).all().item()))
        self.assertTrue(bool(torch.isfinite(candidate.grad).all().item()))


if __name__ == "__main__":
    unittest.main()
