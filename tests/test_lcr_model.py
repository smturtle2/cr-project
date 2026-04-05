from __future__ import annotations

import unittest

import torch

from modules.loss_fn import LCRLoss
from modules.model.lcr import LCR


class LCRModelTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_forward_returns_full_resolution_prediction(self) -> None:
        model = LCR(dim=32, num_blocks=2, heads=4, window_size=8)

        sar = torch.randn(2, 2, 256, 256)
        cloudy = torch.randn(2, 13, 256, 256)

        prediction = model(sar, cloudy)

        self.assertEqual(prediction.shape, (2, 13, 256, 256))

    def test_forward_handles_non_window_aligned_spatial_size(self) -> None:
        model = LCR(dim=32, num_blocks=1, heads=4, window_size=8)

        sar = torch.randn(1, 2, 238, 250)
        cloudy = torch.randn(1, 13, 238, 250)

        prediction = model(sar, cloudy)

        self.assertEqual(prediction.shape, (1, 13, 238, 250))

    def test_mismatched_spatial_size_raises(self) -> None:
        model = LCR(dim=32, num_blocks=1, heads=4, window_size=8)

        sar = torch.randn(1, 2, 256, 256)
        cloudy = torch.randn(1, 13, 255, 256)

        with self.assertRaisesRegex(ValueError, "same batch and spatial size"):
            model(sar, cloudy)

    def test_lcr_loss_backward_is_finite(self) -> None:
        model = LCR(dim=32, num_blocks=1, heads=4, window_size=8)
        loss_fn = LCRLoss()

        sar = torch.randn(1, 2, 64, 64)
        cloudy = torch.randn(1, 13, 64, 64)
        target = torch.randn(1, 13, 64, 64)

        prediction = model(sar, cloudy)
        loss = loss_fn(prediction, target)
        loss.backward()

        self.assertTrue(bool(torch.isfinite(loss).item()))
        grads = [param.grad for param in model.parameters() if param.requires_grad]
        self.assertTrue(
            any(
                grad is not None
                and bool(torch.isfinite(grad).all().item())
                and float(grad.abs().sum()) > 0.0
                for grad in grads
            )
        )


if __name__ == "__main__":
    unittest.main()
