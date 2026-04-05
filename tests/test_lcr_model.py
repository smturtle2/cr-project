from __future__ import annotations

import inspect
import unittest

import torch

from modules.loss_fn import LCRLoss
from modules.model.lcr import LCR
from modules.model.lcr.model import GlobalBlock, NeighborhoodCrossAttention, ShuffleUp2x


class LCRModelTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_forward_returns_full_resolution_prediction(self) -> None:
        model = LCR(dim=32, num_blocks=2, heads=4, neighborhood_size=7)

        sar = torch.randn(2, 2, 256, 256)
        cloudy = torch.randn(2, 13, 256, 256)

        prediction = model(sar, cloudy)

        self.assertEqual(prediction.shape, (2, 13, 256, 256))

    def test_forward_handles_non_window_aligned_spatial_size(self) -> None:
        model = LCR(dim=32, num_blocks=1, heads=4, neighborhood_size=7)

        sar = torch.randn(1, 2, 244, 260)
        cloudy = torch.randn(1, 13, 244, 260)

        prediction = model(sar, cloudy)

        self.assertEqual(prediction.shape, (1, 13, 244, 260))

    def test_mismatched_spatial_size_raises(self) -> None:
        model = LCR(dim=32, num_blocks=1, heads=4, neighborhood_size=7)

        sar = torch.randn(1, 2, 256, 256)
        cloudy = torch.randn(1, 13, 255, 256)

        with self.assertRaisesRegex(ValueError, "same batch and spatial size"):
            model(sar, cloudy)

    def test_lcr_loss_backward_is_finite(self) -> None:
        model = LCR(dim=32, num_blocks=1, heads=4, neighborhood_size=7)
        loss_fn = LCRLoss()

        sar = torch.randn(1, 2, 64, 64)
        cloudy = torch.randn(1, 13, 64, 64)
        target = torch.randn(1, 13, 64, 64)

        prediction = model(sar, cloudy)
        loss = loss_fn(prediction, target)
        loss.backward()

        self.assertTrue(bool(torch.isfinite(loss).item()))
        grads = [param.grad for param in model.parameters() if param.requires_grad]
        self.assertTrue(
            any(
                grad is not None
                and bool(torch.isfinite(grad).all().item())
                and float(grad.abs().sum()) > 0.0
                for grad in grads
            )
        )

    def test_model_uses_patchify_stem_and_pixel_shuffle_decoder(self) -> None:
        model = LCR(dim=32, num_blocks=1, heads=4, neighborhood_size=7)

        self.assertIsInstance(model.sar_stem.down1, torch.nn.Conv2d)
        self.assertIsInstance(model.sar_stem.down2, torch.nn.Conv2d)
        self.assertIsInstance(model.hsi_stem.down1, torch.nn.Conv2d)
        self.assertIsInstance(model.hsi_stem.down2, torch.nn.Conv2d)
        self.assertEqual(model.sar_stem.down1.kernel_size, (2, 2))
        self.assertEqual(model.sar_stem.down1.stride, (2, 2))
        self.assertEqual(model.sar_stem.down1.padding, (0, 0))
        self.assertEqual(model.sar_stem.down2.kernel_size, (2, 2))
        self.assertEqual(model.sar_stem.down2.stride, (2, 2))
        self.assertEqual(model.sar_stem.down2.padding, (0, 0))
        self.assertEqual(model.hsi_stem.down1.kernel_size, (2, 2))
        self.assertEqual(model.hsi_stem.down1.stride, (2, 2))
        self.assertEqual(model.hsi_stem.down1.padding, (0, 0))
        self.assertEqual(model.hsi_stem.down2.kernel_size, (2, 2))
        self.assertEqual(model.hsi_stem.down2.stride, (2, 2))
        self.assertEqual(model.hsi_stem.down2.padding, (0, 0))
        self.assertIsInstance(model.local_blocks[0].attn, NeighborhoodCrossAttention)
        self.assertIsInstance(model.reconstruction.upsample_h2, ShuffleUp2x)
        self.assertIsInstance(model.reconstruction.upsample_full, ShuffleUp2x)
        self.assertIsInstance(model.reconstruction.upsample_h2.shuffle, torch.nn.PixelShuffle)
        self.assertEqual(model.reconstruction.upsample_h2.fuse.kernel_size, (1, 1))
        self.assertIsInstance(model.reconstruction.upsample_full.shuffle, torch.nn.PixelShuffle)
        self.assertEqual(model.reconstruction.upsample_full.fuse.kernel_size, (1, 1))
        self.assertEqual(model.reconstruction.proj_full.kernel_size, (1, 1))
        self.assertEqual(model.reconstruction.out.kernel_size, (1, 1))
        self.assertEqual(model.candidate_head[0].kernel_size, (1, 1))
        self.assertEqual(model.candidate_head[2].kernel_size, (1, 1))
        self.assertEqual(model.mask_head_coarse[0].kernel_size, (1, 1))
        self.assertEqual(model.mask_head_coarse[2].kernel_size, (1, 1))
        self.assertEqual(model.mask_head_refine[0].kernel_size, (1, 1))
        self.assertEqual(model.mask_head_refine[2].kernel_size, (1, 1))

    def test_lcr_contains_no_3x3_convolutions(self) -> None:
        model = LCR(dim=32, num_blocks=1, heads=4, neighborhood_size=7)

        kernel_sizes = [
            module.kernel_size
            for module in model.modules()
            if isinstance(module, torch.nn.Conv2d)
        ]
        self.assertTrue(kernel_sizes)
        self.assertFalse(any(kernel_size == (3, 3) for kernel_size in kernel_sizes))

    def test_global_block_uses_max_pooling(self) -> None:
        source = inspect.getsource(GlobalBlock.forward)

        self.assertIn("max_pool2d", source)
        self.assertNotIn("avg_pool2d", source)

    def test_pixel_shuffle_upsample_is_not_exact_2x2_copy(self) -> None:
        torch.manual_seed(0)
        upsample = ShuffleUp2x(8)
        x = torch.ones(1, 8, 2, 2)

        y = upsample(x)

        top_left_tile = y[0, 0, :2, :2]
        self.assertFalse(torch.allclose(top_left_tile, top_left_tile[0, 0].expand_as(top_left_tile)))

    def test_even_neighborhood_size_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive odd integer"):
            LCR(dim=32, num_blocks=1, heads=4, neighborhood_size=8)

    def test_non_positive_neighborhood_size_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive odd integer"):
            LCR(dim=32, num_blocks=1, heads=4, neighborhood_size=0)

    def test_window_size_keyword_is_no_longer_supported(self) -> None:
        with self.assertRaises(TypeError):
            LCR(dim=32, num_blocks=1, heads=4, window_size=7)


if __name__ == "__main__":
    unittest.main()
