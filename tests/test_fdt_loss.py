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

        self.assertGreater(float(loss), 4.0 - 1e-4)

    def test_feature_scores_are_computed_per_sample(self) -> None:
        loss_fn = FDTDecompositionLoss()
        first = torch.randn(1, 16, 16, 16)
        second = torch.randn_like(first)
        unrelated = torch.randn_like(first)
        sar_com = torch.cat((first, second), dim=0)
        cld_com = torch.cat((first, unrelated), dim=0)

        channel_score, spatial_score = loss_fn._feature_scores(sar_com, cld_com)

        self.assertEqual(channel_score.shape, (2,))
        self.assertEqual(spatial_score.shape, (2,))
        self.assertGreater(float(channel_score[0]), 0.999)
        self.assertGreater(float(spatial_score[0]), 0.999)
        self.assertLess(abs(float(channel_score[1])), 0.1)
        self.assertLess(abs(float(spatial_score[1])), 0.1)

    def test_common_loss_penalizes_global_scale_mismatch(self) -> None:
        loss_fn = FDTDecompositionLoss()
        feature = torch.randn(1, 16, 8, 8)

        exact_loss = loss_fn._common_alignment_loss(feature, feature)
        scaled_loss = loss_fn._common_alignment_loss(feature, 5.0 * feature)

        self.assertGreater(float(scaled_loss), float(exact_loss) + 0.7)

    def test_common_loss_penalizes_spatial_distribution_mismatch(self) -> None:
        loss_fn = FDTDecompositionLoss()
        sar_com = torch.randn(1, 16, 16, 16)
        cld_com = sar_com.roll(shifts=5, dims=-1)

        loss = loss_fn._common_alignment_loss(sar_com, cld_com)

        self.assertGreater(float(loss), 1.8)

    def test_common_loss_penalizes_channel_distribution_mismatch(self) -> None:
        loss_fn = FDTDecompositionLoss()
        sar_com = torch.randn(1, 16, 16, 16)
        cld_com = sar_com.roll(shifts=3, dims=1)

        loss = loss_fn._common_alignment_loss(sar_com, cld_com)

        self.assertGreater(float(loss), 1.8)

    def test_feature_scores_require_same_axis_patterns(self) -> None:
        loss_fn = FDTDecompositionLoss()
        feature = torch.randn(1, 32, 16, 16)
        spatial_permuted = feature.roll(shifts=5, dims=-1)
        channel_permuted = feature.roll(shifts=7, dims=1)
        both_permuted = channel_permuted.roll(shifts=5, dims=-1)

        exact_ch, exact_sp = loss_fn._feature_scores(feature, feature)
        spatial_ch, spatial_sp = loss_fn._feature_scores(feature, spatial_permuted)
        channel_ch, channel_sp = loss_fn._feature_scores(feature, channel_permuted)
        both_ch, both_sp = loss_fn._feature_scores(feature, both_permuted)

        self.assertGreater(float(exact_ch), 0.999)
        self.assertGreater(float(exact_sp), 0.999)
        for score in (spatial_ch, spatial_sp, channel_ch, channel_sp, both_ch, both_sp):
            self.assertLess(abs(float(score)), 0.1)

    def test_independent_dense_features_have_low_scores(self) -> None:
        loss_fn = FDTDecompositionLoss()
        first = torch.randn(8, 32, 32, 32)
        second = torch.randn_like(first)

        channel_score, spatial_score = loss_fn._feature_scores(first, second)

        self.assertLess(float(channel_score.abs().mean()), 0.1)
        self.assertLess(float(spatial_score.abs().mean()), 0.1)

    def test_all_active_feature_has_low_common_scores(self) -> None:
        loss_fn = FDTDecompositionLoss()
        feature = torch.zeros(1, 4, 8, 8)
        feature[:, 0, 2, 2] = 1.0
        all_active = torch.ones_like(feature)

        channel_score, spatial_score = loss_fn._feature_scores(feature, all_active)
        loss = loss_fn._common_alignment_loss(feature, all_active)

        self.assertLess(float(channel_score.abs()), 1e-4)
        self.assertLess(float(spatial_score.abs()), 1e-4)
        self.assertGreater(float(loss), 1.99)
        self.assertLess(float(loss), 2.01)

    def test_comp_loss_is_computed_per_sample(self) -> None:
        loss_fn = FDTDecompositionLoss()
        first = torch.randn(1, 16, 16, 16)
        second = torch.randn_like(first)
        unrelated = torch.randn_like(first)
        sar_comp = torch.cat((first, second), dim=0)
        cld_comp = torch.cat((first, unrelated), dim=0)

        loss = loss_fn._comp_decorrelation_loss(sar_comp, cld_comp)

        self.assertAlmostEqual(float(loss), 1.0, delta=0.05)

    def test_comp_loss_penalizes_axis_score_cancellation(self) -> None:
        loss_fn = FDTDecompositionLoss()
        first = torch.tensor([[[[1.0, 1.0]], [[-1.0, -1.0]]]])
        second = torch.tensor([[[[1.0, -1.0]], [[-1.0, 1.0]]]])

        channel_score, _ = loss_fn._feature_scores(first, second)
        loss = loss_fn._comp_decorrelation_loss(first, second)

        self.assertLess(abs(float(channel_score)), 1e-4)
        self.assertGreater(float(loss), 0.99)

    def test_comp_loss_penalizes_matching_raw_features(self) -> None:
        loss_fn = FDTDecompositionLoss()
        sar_comp = torch.randn(1, 16, 16, 16)
        cld_comp = sar_comp.clone()
        unrelated_comp = torch.randn_like(sar_comp)

        correlated_loss = loss_fn._comp_decorrelation_loss(sar_comp, cld_comp)
        unrelated_loss = loss_fn._comp_decorrelation_loss(sar_comp, unrelated_comp)

        self.assertGreater(float(correlated_loss), float(unrelated_loss) + 1.9)

    def test_comp_loss_is_low_when_channel_and_spatial_patterns_differ(self) -> None:
        loss_fn = FDTDecompositionLoss()
        sar_comp = torch.randn(1, 32, 16, 16)
        channel_permuted = sar_comp.roll(shifts=7, dims=1)
        spatial_permuted = sar_comp.roll(shifts=5, dims=-1)

        channel_loss = loss_fn._comp_decorrelation_loss(sar_comp, channel_permuted)
        spatial_loss = loss_fn._comp_decorrelation_loss(sar_comp, spatial_permuted)

        self.assertLess(float(channel_loss), 0.1)
        self.assertLess(float(spatial_loss), 0.1)

    def test_comp_loss_penalizes_positive_and_negative_raw_match(self) -> None:
        loss_fn = FDTDecompositionLoss()
        feature = torch.randn(1, 16, 16, 16)
        unrelated_feature = torch.randn_like(feature)

        same_loss = loss_fn._comp_decorrelation_loss(feature, feature)
        opposite_loss = loss_fn._comp_decorrelation_loss(feature, -feature)
        unrelated_loss = loss_fn._comp_decorrelation_loss(feature, unrelated_feature)

        self.assertGreater(float(same_loss), float(unrelated_loss) + 1.9)
        self.assertGreater(float(opposite_loss), float(unrelated_loss) + 1.9)

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
