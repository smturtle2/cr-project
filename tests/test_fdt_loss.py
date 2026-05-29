from __future__ import annotations

import unittest

import torch

from modules.loss_fn import (
    FDTCCALoss,
    FDTDecompositionLoss,
    FeatureUncorrelationLoss,
    PatchSlicedWassersteinLoss,
)


class PatchSlicedWassersteinLossTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_identical_features_have_zero_loss(self) -> None:
        loss_fn = PatchSlicedWassersteinLoss(num_projections=32)
        feature = torch.randn(2, 16, 8, 8)

        loss = loss_fn(feature, feature)

        self.assertLess(float(loss), 1e-6)

    def test_shifted_distribution_has_larger_loss(self) -> None:
        loss_fn = PatchSlicedWassersteinLoss(num_projections=64)
        feature = torch.randn(2, 16, 8, 8)

        exact_loss = loss_fn(feature, feature)
        shifted_loss = loss_fn(feature, feature + 2.0)

        self.assertGreater(float(shifted_loss), float(exact_loss) + 1.0)

    def test_texture_distribution_mismatch_has_larger_loss(self) -> None:
        loss_fn = PatchSlicedWassersteinLoss(num_projections=64)
        axis = torch.linspace(-1.0, 1.0, 16)
        smooth = axis.view(1, 1, 1, 16) + axis.view(1, 1, 16, 1)
        checker = (
            (torch.arange(16).view(16, 1) + torch.arange(16).view(1, 16)) % 2
        ).float()
        checker = checker.mul(2.0).sub(1.0).view(1, 1, 16, 16)
        smooth = smooth.repeat(1, 8, 1, 1)
        checker = checker.repeat(1, 8, 1, 1)
        smooth = (smooth - smooth.mean()) / smooth.std()
        checker = (checker - checker.mean()) / checker.std()

        exact_loss = loss_fn(smooth, smooth)
        texture_loss = loss_fn(smooth, checker)

        self.assertGreater(float(texture_loss), float(exact_loss) + 0.2)


class FeatureUncorrelationLossTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_matching_features_have_large_loss(self) -> None:
        loss_fn = FeatureUncorrelationLoss()
        feature = torch.randn(1, 16, 16, 16)

        loss = loss_fn(feature, feature)

        self.assertGreater(float(loss), 1.9)

    def test_opposite_features_have_large_loss(self) -> None:
        loss_fn = FeatureUncorrelationLoss()
        feature = torch.randn(1, 16, 16, 16)

        loss = loss_fn(feature, -feature)

        self.assertGreater(float(loss), 1.9)

    def test_unrelated_features_have_small_loss(self) -> None:
        loss_fn = FeatureUncorrelationLoss()
        first = torch.randn(2, 32, 16, 16)
        second = torch.randn_like(first)

        loss = loss_fn(first, second)

        self.assertLess(float(loss), 0.1)


class FDTDecompositionLossTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_backward_is_finite(self) -> None:
        loss_fn = FDTDecompositionLoss()
        candidate = torch.randn(2, 13, 16, 16, requires_grad=True)
        target = torch.randn(2, 13, 16, 16)

        loss = loss_fn(candidate, target)
        loss.backward()

        self.assertTrue(bool(torch.isfinite(loss).item()))
        self.assertIsNotNone(candidate.grad)
        self.assertTrue(bool(torch.isfinite(candidate.grad).all().item()))

    def test_matches_candidate_l1_loss(self) -> None:
        loss_fn = FDTDecompositionLoss()
        candidate = torch.randn(2, 13, 16, 16)
        target = torch.randn(2, 13, 16, 16)

        loss = loss_fn(candidate, target)

        self.assertTrue(torch.allclose(loss, torch.mean(torch.abs(candidate - target))))


class FDTCCALossTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_combines_prediction_ssim_and_candidate_losses(self) -> None:
        loss_fn = FDTCCALoss(ssim_weight=0.5, candidate_loss_weight=0.1)
        prediction = torch.randn(2, 13, 16, 16)
        candidate = torch.randn(2, 13, 16, 16)
        target = torch.randn(2, 13, 16, 16)
        model_output = (prediction, candidate, None, None, None, None)

        loss = loss_fn(model_output, target)
        expected = (
            torch.mean(torch.abs(prediction - target))
            + 0.5 * (1.0 - loss_fn.ssim(prediction, target))
            + 0.1 * torch.mean(torch.abs(candidate - target))
        )

        self.assertTrue(torch.allclose(loss, expected))

    def test_accepts_prediction_tensor_when_candidate_loss_is_disabled(self) -> None:
        loss_fn = FDTCCALoss(ssim_weight=0.5, candidate_loss_weight=0.0)
        prediction = torch.randn(2, 13, 16, 16)
        target = torch.randn(2, 13, 16, 16)

        loss = loss_fn(prediction, target)
        expected = torch.mean(torch.abs(prediction - target)) + 0.5 * (
            1.0 - loss_fn.ssim(prediction, target)
        )

        self.assertTrue(torch.allclose(loss, expected))

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
