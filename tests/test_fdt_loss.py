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

        loss = loss_fn._common_alignment_loss(feature, feature)

        self.assertLess(float(loss), 1e-4)

    def test_sign_flipped_common_features_have_large_common_loss(self) -> None:
        loss_fn = FDTDecompositionLoss()
        feature = torch.randn(2, 32, 16, 16)

        loss = loss_fn._common_alignment_loss(feature, -feature)

        self.assertGreater(float(loss), 1.0 - 1e-4)

    def test_feature_score_is_computed_per_sample(self) -> None:
        loss_fn = FDTDecompositionLoss()
        first = torch.zeros(1, 4, 8, 8)
        first[:, 0, 2, 2] = 1.0
        second = torch.zeros_like(first)
        second[:, 1, 5, 5] = 1.0
        sar_com = torch.cat((first, second), dim=0)
        cld_com = torch.cat((first, first), dim=0)

        score = loss_fn._feature_score(sar_com, cld_com)

        self.assertEqual(score.shape, (2,))
        self.assertGreater(float(score[0]), 0.999)
        self.assertLess(abs(float(score[1])), 1e-2)

    def test_common_loss_penalizes_global_scale_mismatch(self) -> None:
        loss_fn = FDTDecompositionLoss()
        feature = torch.randn(1, 16, 8, 8)

        exact_loss = loss_fn._common_alignment_loss(feature, feature)
        scaled_loss = loss_fn._common_alignment_loss(feature, 5.0 * feature)

        self.assertGreater(float(scaled_loss), float(exact_loss) + 0.25)

    def test_common_loss_penalizes_spatial_distribution_mismatch(self) -> None:
        loss_fn = FDTDecompositionLoss()
        sar_com = torch.zeros(1, 4, 8, 8)
        cld_com = torch.zeros_like(sar_com)
        sar_com[:, 0, 2, 2] = 1.0
        cld_com[:, 0, 5, 5] = 1.0

        loss = loss_fn._common_alignment_loss(sar_com, cld_com)

        self.assertGreater(float(loss), 0.45)

    def test_common_loss_penalizes_channel_distribution_mismatch(self) -> None:
        loss_fn = FDTDecompositionLoss()
        sar_com = torch.zeros(1, 4, 8, 8)
        cld_com = torch.zeros_like(sar_com)
        sar_com[:, 0, 2, 2] = 1.0
        cld_com[:, 1, 2, 2] = 1.0

        loss = loss_fn._common_alignment_loss(sar_com, cld_com)

        self.assertGreater(float(loss), 0.45)

    def test_feature_score_requires_same_joint_location(self) -> None:
        loss_fn = FDTDecompositionLoss()
        feature = torch.zeros(1, 4, 8, 8)
        feature[:, 0, 2, 2] = 1.0
        spatial_permuted = torch.zeros_like(feature)
        spatial_permuted[:, 0, 5, 5] = 1.0
        channel_permuted = torch.zeros_like(feature)
        channel_permuted[:, 1, 2, 2] = 1.0
        both_permuted = torch.zeros_like(feature)
        both_permuted[:, 1, 5, 5] = 1.0

        exact_score = loss_fn._feature_score(feature, feature)
        spatial_score = loss_fn._feature_score(feature, spatial_permuted)
        channel_score = loss_fn._feature_score(feature, channel_permuted)
        both_score = loss_fn._feature_score(feature, both_permuted)

        self.assertGreater(float(exact_score), 0.999)
        self.assertLess(abs(float(spatial_score)), 1e-2)
        self.assertLess(abs(float(channel_score)), 1e-2)
        self.assertLess(abs(float(both_score)), 1e-2)

    def test_independent_dense_features_have_low_score(self) -> None:
        loss_fn = FDTDecompositionLoss()
        first = torch.randn(8, 32, 32, 32)
        second = torch.randn_like(first)

        score = loss_fn._feature_score(first, second)

        self.assertLess(float(score.abs().mean()), 0.1)

    def test_all_active_feature_has_low_common_score(self) -> None:
        loss_fn = FDTDecompositionLoss()
        feature = torch.zeros(1, 4, 8, 8)
        feature[:, 0, 2, 2] = 1.0
        all_active = torch.ones_like(feature)

        score = loss_fn._feature_score(feature, all_active)
        loss = loss_fn._common_alignment_loss(feature, all_active)

        self.assertLess(float(score.abs()), 1e-4)
        self.assertGreater(float(loss), 0.49)
        self.assertLess(float(loss), 0.51)

    def test_comp_loss_is_computed_per_sample(self) -> None:
        loss_fn = FDTDecompositionLoss()
        first = torch.zeros(1, 16, 8, 8)
        first[:, 0, 2, 2] = 1.0
        second = torch.zeros_like(first)
        second[:, 1, 5, 5] = 1.0
        sar_comp = torch.cat((first, second), dim=0)
        cld_comp = torch.cat((first, first), dim=0)

        loss = loss_fn._comp_decorrelation_loss(sar_comp, cld_comp)

        self.assertAlmostEqual(float(loss), 0.5, delta=1e-3)

    def test_comp_loss_penalizes_matching_raw_features(self) -> None:
        loss_fn = FDTDecompositionLoss()
        sar_comp = torch.zeros(1, 4, 8, 8)
        sar_comp[:, 0, 2, 2] = 1.0
        cld_comp = sar_comp.clone()
        unrelated_comp = torch.zeros_like(sar_comp)
        unrelated_comp[:, 1, 5, 5] = 1.0

        correlated_loss = loss_fn._comp_decorrelation_loss(sar_comp, cld_comp)
        unrelated_loss = loss_fn._comp_decorrelation_loss(sar_comp, unrelated_comp)

        self.assertGreater(float(correlated_loss), float(unrelated_loss) + 0.9)

    def test_comp_loss_ignores_shared_axis_when_joint_location_differs(self) -> None:
        loss_fn = FDTDecompositionLoss()
        sar_comp = torch.zeros(1, 4, 8, 8)
        cld_comp = torch.zeros_like(sar_comp)
        unrelated_comp = torch.zeros_like(sar_comp)
        sar_comp[:, 0, 2, 2] = 1.0
        cld_comp[:, 1, 2, 2] = 1.0
        unrelated_comp[:, 1, 5, 5] = 1.0

        correlated_loss = loss_fn._comp_decorrelation_loss(sar_comp, cld_comp)
        unrelated_loss = loss_fn._comp_decorrelation_loss(sar_comp, unrelated_comp)

        self.assertLess(float(correlated_loss), 1e-4)
        self.assertLess(float(unrelated_loss), 1e-4)

    def test_comp_loss_penalizes_positive_and_negative_raw_match(self) -> None:
        loss_fn = FDTDecompositionLoss()
        feature = torch.zeros(1, 4, 8, 8)
        feature[:, 0, 2, 2] = 1.0
        unrelated_feature = torch.zeros_like(feature)
        unrelated_feature[:, 1, 5, 5] = 1.0

        same_loss = loss_fn._comp_decorrelation_loss(feature, feature)
        opposite_loss = loss_fn._comp_decorrelation_loss(feature, -feature)
        unrelated_loss = loss_fn._comp_decorrelation_loss(feature, unrelated_feature)

        self.assertGreater(float(same_loss), float(unrelated_loss) + 0.9)
        self.assertGreater(float(opposite_loss), float(unrelated_loss) + 0.9)

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
