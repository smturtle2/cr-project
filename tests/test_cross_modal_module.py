from __future__ import annotations

import unittest

import torch

from modules.model.baseline.ACA_CRNet import ACA_CRNet
from modules.model.baseline.ca_flash import ConAttn as FlashConAttn
from modules.model.module.base_module import BaseModule
from modules.model.module.cross_attention_module import CrossAttentionModule


class CrossModalModuleTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_cross_attention_module_returns_updated_feature(self) -> None:
        module = CrossAttentionModule(
            sar_channels=2,
            feature_channels=16,
            num_heads=4,
            self_num_layers=1,
        )

        sar = torch.randn(2, 2, 16, 16)
        feature = torch.randn(2, 16, 16, 16)

        updated = module(sar, feature)

        self.assertEqual(updated.shape, feature.shape)

    def test_base_module_uses_cross_modal_without_mask(self) -> None:
        module = BaseModule(
            sar_channels=2,
            cloudy_channels=13,
            feature_channels=16,
            num_heads=4,
            self_num_layers=1,
        )

        sar = torch.randn(1, 2, 16, 16)
        cloudy = torch.randn(1, 13, 16, 16)
        feature = torch.randn(1, 16, 16, 16)

        updated = module(sar, cloudy, feature)

        self.assertEqual(updated.shape, feature.shape)
        self.assertFalse(hasattr(module, "mask"))

    def test_aca_crnet_forward_uses_cross_modal_base_module(self) -> None:
        model = ACA_CRNet(
            in_channels=15,
            out_channels=13,
            num_layers=2,
            feature_sizes=16,
            ca=FlashConAttn,
        )

        sar = torch.randn(1, 2, 16, 16)
        cloudy = torch.randn(1, 13, 16, 16)

        prediction = model(sar, cloudy)

        self.assertEqual(prediction.shape, (1, 13, 16, 16))

    def test_cross_modal_backward_reaches_sar_encoder_and_block(self) -> None:
        module = CrossAttentionModule(
            sar_channels=2,
            feature_channels=16,
            num_heads=4,
            self_num_layers=1,
        )

        sar = torch.randn(1, 2, 16, 16)
        feature = torch.randn(1, 16, 16, 16)

        loss = module(sar, feature).abs().mean()
        loss.backward()

        sar_grad = module.sar_encoder.net[0].weight.grad
        attn_grad = module.cross_modal.attn.project_out.weight.grad
        ffn_grad = module.cross_modal.ffn.proj_out.weight.grad

        self.assertIsNotNone(sar_grad)
        self.assertIsNotNone(attn_grad)
        self.assertIsNotNone(ffn_grad)


if __name__ == "__main__":
    unittest.main()
