from __future__ import annotations

import unittest

import torch

from modules.loss_fn import FDTDecompositionLoss


class FDTDecompositionLossTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_identical_common_features_have_small_common_loss(self) -> None:
        loss_fn = FDTDecompositionLoss()
        feature = torch.randn(1, 32, 16, 16)
        zeros = torch.zeros_like(feature)

        loss = loss_fn(feature, feature, zeros, zeros)

        self.assertLess(float(loss), 1e-4)

    def test_negative_common_correlation_has_large_loss(self) -> None:
        loss_fn = FDTDecompositionLoss()
        feature = torch.randn(2, 32, 16, 16)
        zeros = torch.zeros_like(feature)

        loss = loss_fn(feature, -feature, zeros, zeros)

        self.assertGreater(float(loss), 1.9)

    def test_common_loss_is_computed_per_sample(self) -> None:
        loss_fn = FDTDecompositionLoss()
        first = torch.randn(1, 16, 8, 8)
        second = torch.randn(1, 16, 8, 8)
        sar_com = torch.cat((first, second), dim=0)
        cld_com = torch.cat((first, second), dim=0)
        zeros = torch.zeros_like(sar_com)

        loss = loss_fn(sar_com, cld_com, zeros, zeros)

        self.assertLess(float(loss), 1e-4)

    def test_comp_loss_is_computed_per_sample(self) -> None:
        loss_fn = FDTDecompositionLoss()
        common = torch.randn(2, 16, 8, 8)
        first = torch.randn(1, 16, 8, 8)
        second = torch.randn(1, 16, 8, 8)
        sar_comp = torch.cat((first, second), dim=0)
        cld_comp = torch.cat((first, -second), dim=0)

        loss = loss_fn(common, common, sar_comp, cld_comp)

        self.assertGreater(float(loss), 0.05)

    def test_comp_loss_penalizes_cross_channel_correlation(self) -> None:
        loss_fn = FDTDecompositionLoss()
        common = torch.randn(1, 4, 8, 8)
        first_channel = torch.randn(1, 1, 8, 8)
        sar_comp = torch.cat(
            (
                torch.randn(1, 1, 8, 8),
                first_channel,
                torch.randn(1, 2, 8, 8),
            ),
            dim=1,
        )
        cld_comp = torch.cat(
            (
                torch.randn(1, 3, 8, 8),
                first_channel,
            ),
            dim=1,
        )
        unrelated_comp = torch.randn_like(cld_comp)

        correlated_loss = loss_fn(common, common, sar_comp, cld_comp)
        unrelated_loss = loss_fn(common, common, sar_comp, unrelated_comp)

        self.assertGreater(float(correlated_loss), float(unrelated_loss))

    def test_comp_loss_penalizes_positive_and_negative_correlation(self) -> None:
        loss_fn = FDTDecompositionLoss()
        common = torch.randn(1, 96, 32, 32)
        feature = torch.randn_like(common)
        unrelated_feature = torch.randn_like(feature)

        same_loss = loss_fn(common, common, feature, feature)
        opposite_loss = loss_fn(common, common, feature, -feature)
        unrelated_loss = loss_fn(common, common, feature, unrelated_feature)

        self.assertGreater(float(same_loss), float(unrelated_loss))
        self.assertGreater(float(opposite_loss), float(unrelated_loss))

    def test_constant_inputs_are_finite(self) -> None:
        loss_fn = FDTDecompositionLoss()
        sar_com = torch.ones(1, 16, 8, 8, requires_grad=True)
        cld_com = torch.ones(1, 16, 8, 8, requires_grad=True)
        sar_comp = torch.ones(1, 16, 8, 8, requires_grad=True)
        cld_comp = torch.ones(1, 16, 8, 8, requires_grad=True)

        loss = loss_fn(sar_com, cld_com, sar_comp, cld_comp)
        loss.backward()

        self.assertTrue(bool(torch.isfinite(loss).item()))
        for feature in (sar_com, cld_com, sar_comp, cld_comp):
            self.assertIsNotNone(feature.grad)
            self.assertTrue(bool(torch.isfinite(feature.grad).all().item()))

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
