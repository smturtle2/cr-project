from __future__ import annotations

import unittest

import torch
from torch import nn

from modules.metrics.gate_eval import prepare_gate_for_eval, summarize_gate
from modules.model.baseline.ACA_CRNet import ACA_CRNet
from modules.model.gate import CosineGateEstimator, CosinePriorGateEstimator


class IdentityCA(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class GateEstimatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sar = torch.rand(2, 2, 16, 16)
        self.cloudy = torch.rand(2, 13, 16, 16)
        self.target = torch.rand(2, 13, 16, 16)

    def test_estimators_return_single_channel_gate_in_unit_interval(self) -> None:
        estimators = [
            CosineGateEstimator(feat_dim=8),
            CosinePriorGateEstimator(feat_dim=8),
        ]
        for estimator in estimators:
            with self.subTest(estimator=type(estimator).__name__):
                gate = estimator(self.sar, self.cloudy)
                self.assertEqual(gate.shape, (2, 1, 16, 16))
                self.assertGreaterEqual(float(gate.min()), 0.0)
                self.assertLessEqual(float(gate.max()), 1.0)

    def test_aca_crnet_wires_all_gate_modes(self) -> None:
        expected_channels = {"mask": 16, "cosine": 1, "cosine_prior": 1}
        for mode, channels in expected_channels.items():
            with self.subTest(mode=mode):
                model = ACA_CRNet(
                    in_channels=15,
                    out_channels=13,
                    num_layers=4,
                    feature_sizes=16,
                    ca=IdentityCA,
                    gate_mode=mode,
                    gate_feat_dim=8,
                )
                prediction = model(self.sar, self.cloudy)
                self.assertEqual(prediction.shape, self.cloudy.shape)
                self.assertIsNotNone(model.last_gate)
                self.assertEqual(model.last_gate.shape, (2, channels, 16, 16))
                self.assertEqual(len(model.last_gates), 2)

    def test_gate_summary_accepts_multichannel_mask_gate(self) -> None:
        mask_gate = torch.rand(2, 16, 16, 16)
        prepared = prepare_gate_for_eval(mask_gate, target_hw=(16, 16))
        self.assertEqual(prepared.shape, (2, 1, 16, 16))
        result = summarize_gate(mask_gate, self.cloudy, self.target)
        self.assertGreaterEqual(result.thick_mean_g, 0.0)
        self.assertLessEqual(result.thick_mean_g, 1.0)


if __name__ == "__main__":
    unittest.main()
