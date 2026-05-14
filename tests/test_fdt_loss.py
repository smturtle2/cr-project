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

    def test_sign_flipped_common_features_have_large_common_loss(self) -> None:
        loss_fn = FDTDecompositionLoss()
        feature = torch.randn(2, 32, 16, 16)
        zeros = torch.zeros_like(feature)

        loss = loss_fn(feature, -feature, zeros, zeros)

        self.assertGreater(float(loss), 1.9)

    def test_feature_ccc_is_computed_per_sample(self) -> None:
        loss_fn = FDTDecompositionLoss()
        first = torch.randn(1, 16, 8, 8)
        second = torch.randn_like(first)
        sar_com = torch.cat((first, second), dim=0)
        cld_com = torch.cat((first, -second), dim=0)

        ccc = loss_fn._feature_ccc(sar_com, cld_com)

        self.assertEqual(ccc.shape, (2,))
        self.assertGreater(float(ccc[0]), 1.0 - 1e-4)
        self.assertLess(float(ccc[1]), -1.0 + 1e-4)

    def test_common_loss_penalizes_scale_mismatch(self) -> None:
        loss_fn = FDTDecompositionLoss()
        feature = torch.randn(1, 16, 8, 8)
        zeros = torch.zeros_like(feature)

        exact_loss = loss_fn(feature, feature, zeros, zeros)
        scaled_loss = loss_fn(feature, 2.0 * feature, zeros, zeros)

        self.assertGreater(float(scaled_loss), float(exact_loss) + 0.1)

    def test_common_loss_does_not_let_one_dominant_channel_hide_mismatch(self) -> None:
        loss_fn = FDTDecompositionLoss()
        spatial = torch.randn(1, 1, 8, 8)
        mismatch = torch.randn(1, 1, 8, 8)
        sar_com = torch.cat((8.0 * spatial, mismatch), dim=1)
        cld_com = torch.cat((8.0 * spatial, -mismatch), dim=1)
        zeros = torch.zeros_like(sar_com)

        loss = loss_fn(sar_com, cld_com, zeros, zeros)

        self.assertGreater(float(loss), 0.25)

    def test_feature_ccc_balances_channel_and_spatial_axes(self) -> None:
        loss_fn = FDTDecompositionLoss()
        feature = torch.randn(1, 4, 8, 8)
        spatial_permuted = feature.roll(shifts=1, dims=3)
        channel_permuted = feature.roll(shifts=1, dims=1)

        exact_score = loss_fn._feature_ccc(feature, feature)
        spatial_score = loss_fn._feature_ccc(feature, spatial_permuted)
        channel_score = loss_fn._feature_ccc(feature, channel_permuted)

        self.assertGreater(float(exact_score), 1.0 - 1e-4)
        self.assertLess(float(spatial_score), float(exact_score) - 0.1)
        self.assertLess(float(channel_score), float(exact_score) - 0.1)

    def test_comp_loss_is_computed_per_sample(self) -> None:
        loss_fn = FDTDecompositionLoss()
        common = torch.randn(2, 16, 8, 8)
        first = torch.randn(1, 16, 8, 8)
        second = torch.randn(1, 16, 8, 8)
        sar_comp = torch.cat((first, second), dim=0)
        cld_comp = torch.cat((first, -second), dim=0)

        loss = loss_fn(common, common, sar_comp, cld_comp)

        self.assertGreater(float(loss), 0.05)

    def test_comp_loss_penalizes_matching_feature_ccc(self) -> None:
        loss_fn = FDTDecompositionLoss()
        common = torch.randn(1, 4, 8, 8)
        sar_comp = torch.randn(1, 4, 8, 8)
        cld_comp = sar_comp.clone()
        unrelated_comp = torch.randn_like(sar_comp)

        correlated_loss = loss_fn(common, common, sar_comp, cld_comp)
        unrelated_loss = loss_fn(common, common, sar_comp, unrelated_comp)

        self.assertGreater(float(correlated_loss), float(unrelated_loss))

    def test_comp_loss_penalizes_shared_spatial_representation(self) -> None:
        loss_fn = FDTDecompositionLoss()
        common = torch.randn(1, 2, 8, 8)
        spatial = torch.randn(1, 1, 8, 8)
        mismatch = torch.randn(1, 1, 8, 8)
        sar_comp = torch.cat((8.0 * spatial, mismatch), dim=1)
        cld_comp = torch.cat((8.0 * spatial, -mismatch), dim=1)
        unrelated_comp = torch.randn_like(sar_comp)

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
