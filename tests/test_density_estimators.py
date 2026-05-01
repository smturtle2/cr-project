from __future__ import annotations

import unittest

import torch

from modules.metrics.density_eval import summarize_density
from modules.model.cafm.ACA_CRNet import ACA_CRNet
from modules.model.cafm.density import (
    CosineDensityEstimator,
    CosinePriorDensityEstimator,
)


class DensityEstimatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sar = torch.rand(2, 2, 16, 16)
        self.cloudy = torch.rand(2, 13, 16, 16)
        self.target = torch.rand(2, 13, 16, 16)

    def test_estimators_return_density_map_in_unit_interval(self) -> None:
        estimators = [
            CosineDensityEstimator(feat_dim=8),
            CosinePriorDensityEstimator(feat_dim=8),
        ]

        for estimator in estimators:
            with self.subTest(estimator=type(estimator).__name__):
                density = estimator(self.sar, self.cloudy)
                self.assertEqual(density.shape, (2, 1, 16, 16))
                self.assertGreaterEqual(float(density.min()), 0.0)
                self.assertLessEqual(float(density.max()), 1.0)

    def test_aca_crnet_wires_density_mode_without_changing_interface(self) -> None:
        model = ACA_CRNet(
            num_layers=4,
            feature_sizes=16,
            cafm_feat_dim=8,
            density_mode="cosine_prior",
            use_cafm=True,
            use_sdi=False,
        )
        prediction = model(self.sar, self.cloudy)
        self.assertEqual(prediction.shape, self.cloudy.shape)
        self.assertIsNotNone(model.last_density)
        self.assertEqual(model.last_density.shape, (2, 1, 16, 16))

    def test_density_summary_orders_clear_thin_thick(self) -> None:
        weighted = (self.cloudy - self.target).abs().mean(dim=1, keepdim=True)
        result = summarize_density(weighted, self.cloudy, self.target)
        self.assertGreaterEqual(result.thick_mean_d, result.thin_mean_d)
        self.assertGreaterEqual(result.thin_mean_d, result.clear_mean_d)


if __name__ == "__main__":
    unittest.main()
