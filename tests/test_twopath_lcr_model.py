from __future__ import annotations

import unittest

import torch

from modules.model.twopath import LCR, TwoPathACA_CRNet


class TwoPathLCRModelTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_lcr_mask_net_returns_thirteen_channel_logits(self) -> None:
        model = LCR(
            sar_channels=2,
            opt_channels=13,
            dim=16,
            num_blocks=1,
            heads=4,
            encoder_block_count=1,
        )

        sar = torch.randn(2, 2, 16, 16)
        cloudy = torch.randn(2, 13, 16, 16)

        mask_logits = model(sar, cloudy)

        self.assertEqual(mask_logits.shape, (2, 13, 16, 16))
        self.assertLess(float(mask_logits.min()), 0.0)
        self.assertLess(float(mask_logits.max()), 0.0)

    def test_twopath_lcr_aca_crnet_returns_full_resolution_prediction(self) -> None:
        model = TwoPathACA_CRNet(
            in_channels=15,
            out_channels=13,
            num_layers=1,
            feature_sizes=16,
            lcr_dim=16,
            lcr_num_blocks=1,
            lcr_heads=4,
            lcr_encoder_block_count=1,
        )

        sar = torch.randn(1, 2, 16, 16)
        cloudy = torch.randn(1, 13, 16, 16)

        prediction = model(sar, cloudy)

        self.assertEqual(prediction.shape, (1, 13, 16, 16))

    def test_twopath_lcr_aca_crnet_backward_reaches_both_paths(self) -> None:
        model = TwoPathACA_CRNet(
            in_channels=15,
            out_channels=13,
            num_layers=1,
            feature_sizes=16,
            lcr_dim=16,
            lcr_num_blocks=1,
            lcr_heads=4,
            lcr_encoder_block_count=1,
        )

        sar = torch.randn(1, 2, 16, 16)
        cloudy = torch.randn(1, 13, 16, 16)
        target = torch.randn(1, 13, 16, 16)

        loss = torch.mean(torch.abs(model(sar, cloudy) - target))
        loss.backward()

        self.assertIsNotNone(model.mask_net.mask_head.bias.grad)
        self.assertIsNotNone(model.mask_net.wrapper_blocks[0].cross_blocks[0].attn.q_proj.weight.grad)
        self.assertIsNotNone(model.candidate_net.net[-1].weight.grad)
        self.assertGreater(float(model.mask_net.mask_head.bias.grad.abs().mean()), 0.0)
        self.assertGreater(float(model.candidate_net.net[-1].weight.grad.abs().mean()), 0.0)

    def test_lcr_mask_net_rejects_non_divisible_patch_size(self) -> None:
        model = LCR(
            sar_channels=2,
            opt_channels=13,
            dim=16,
            num_blocks=1,
            heads=4,
            encoder_block_count=1,
            patch_size=4,
        )

        sar = torch.randn(1, 2, 18, 16)
        cloudy = torch.randn(1, 13, 18, 16)

        with self.assertRaisesRegex(ValueError, "divisible by patch_size"):
            model(sar, cloudy)


if __name__ == "__main__":
    unittest.main()
