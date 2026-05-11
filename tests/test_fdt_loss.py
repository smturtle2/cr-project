from __future__ import annotations

import unittest

import torch

from modules.loss_fn import FDTDecompositionLoss


class FDTDecompositionLossTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_identical_common_features_have_small_common_loss(self) -> None:
        loss_fn = FDTDecompositionLoss(comp_weight=0.0)
        feature = torch.randn(1, 96, 32, 32)

        loss = loss_fn(feature, feature, feature, torch.flip(feature, dims=(3,)))

        self.assertLess(float(loss), 1e-4)

    def test_comp_loss_penalizes_positive_and_negative_correlation(self) -> None:
        loss_fn = FDTDecompositionLoss(common_weight=0.0, comp_weight=1.0)
        feature = torch.randn(1, 96, 32, 32)
        unrelated_feature = torch.randn_like(feature)

        same_loss = loss_fn(feature, feature, feature, feature)
        opposite_loss = loss_fn(feature, feature, feature, -feature)
        unrelated_loss = loss_fn(feature, feature, feature, unrelated_feature)

        self.assertGreater(float(same_loss), float(unrelated_loss))
        self.assertGreater(float(opposite_loss), float(unrelated_loss))

    def test_backward_is_finite(self) -> None:
        loss_fn = FDTDecompositionLoss()
        sar_com = torch.randn(2, 96, 32, 32, requires_grad=True)
        cld_com = torch.randn(2, 96, 32, 32, requires_grad=True)
        sar_comp = torch.randn(2, 96, 32, 32, requires_grad=True)
        cld_comp = torch.randn(2, 96, 32, 32, requires_grad=True)

        loss = loss_fn(sar_com, cld_com, sar_comp, cld_comp)
        loss.backward()

        self.assertTrue(bool(torch.isfinite(loss).item()))
        for feature in (sar_com, cld_com, sar_comp, cld_comp):
            self.assertIsNotNone(feature.grad)
            self.assertTrue(bool(torch.isfinite(feature.grad).all().item()))


if __name__ == "__main__":
    unittest.main()
