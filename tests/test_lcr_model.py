from __future__ import annotations

import inspect
import unittest

import torch

from modules.loss_fn import LCRLoss
from modules.model.lcr import model as lcr_model
from modules.model.lcr import LCR
from modules.model.lcr.model import (
    BilinearUp2x,
    GlobalBlock,
    GlobalSelfAttention,
    LCRWrapperBlock,
    LatentDecoder,
    NeighborhoodCrossAttention,
)


class LCRModelTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_forward_returns_full_resolution_prediction(self) -> None:
        model = LCR(
            dim=32,
            num_blocks=2,
            local_block_count=2,
            global_block_count=3,
            heads=4,
            neighborhood_size=7,
        )

        sar = torch.randn(2, 2, 256, 256)
        cloudy = torch.randn(2, 13, 256, 256)

        prediction = model(sar, cloudy)

        self.assertEqual(prediction.shape, (2, 13, 256, 256))

    def test_forward_handles_spatial_size_divisible_by_four_but_not_eight(self) -> None:
        model = LCR(dim=32, num_blocks=1, heads=4, neighborhood_size=7)

        sar = torch.randn(1, 2, 244, 260)
        cloudy = torch.randn(1, 13, 244, 260)

        prediction = model(sar, cloudy)

        self.assertEqual(prediction.shape, (1, 13, 244, 260))

    def test_forward_rejects_spatial_size_not_divisible_by_four(self) -> None:
        model = LCR(dim=32, num_blocks=1, heads=4, neighborhood_size=7)

        sar = torch.randn(1, 2, 246, 262)
        cloudy = torch.randn(1, 13, 246, 262)

        with self.assertRaisesRegex(ValueError, "divisible by 4"):
            model(sar, cloudy)

    def test_mismatched_spatial_size_raises(self) -> None:
        model = LCR(dim=32, num_blocks=1, heads=4, neighborhood_size=7)

        sar = torch.randn(1, 2, 256, 256)
        cloudy = torch.randn(1, 13, 255, 256)

        with self.assertRaisesRegex(ValueError, "same batch and spatial size"):
            model(sar, cloudy)

    def test_lcr_loss_backward_is_finite(self) -> None:
        model = LCR(
            dim=32,
            num_blocks=1,
            local_block_count=2,
            global_block_count=2,
            heads=4,
            neighborhood_size=7,
        )
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

    def test_global_self_attention_parameters_receive_gradients(self) -> None:
        model = LCR(
            dim=32,
            num_blocks=1,
            local_block_count=1,
            global_block_count=1,
            heads=4,
            neighborhood_size=7,
        )
        loss_fn = LCRLoss()

        sar = torch.randn(1, 2, 64, 64)
        cloudy = torch.randn(1, 13, 64, 64)
        target = torch.randn(1, 13, 64, 64)

        prediction = model(sar, cloudy)
        loss = loss_fn(prediction, target)
        loss.backward()

        global_attn = model.wrapper_blocks[0].global_blocks[0].attn
        grad_tensors = [
            global_attn.q_proj.weight.grad,
            global_attn.k_proj.weight.grad,
            global_attn.v_proj.weight.grad,
            global_attn.out_proj.weight.grad,
        ]
        self.assertTrue(
            all(
                grad is not None
                and bool(torch.isfinite(grad).all().item())
                and float(grad.abs().sum()) > 0.0
                for grad in grad_tensors
            )
        )

    def test_model_uses_single_downsample_stem_and_bilinear_decoder(self) -> None:
        model = LCR(
            dim=32,
            num_blocks=1,
            local_block_count=1,
            global_block_count=1,
            heads=4,
            neighborhood_size=7,
        )
        wrapper = model.wrapper_blocks[0]

        self.assertIsInstance(model.sar_stem.downsample, torch.nn.Conv2d)
        self.assertIsInstance(model.hsi_stem.downsample, torch.nn.Conv2d)
        self.assertEqual(model.sar_stem.downsample.kernel_size, (2, 2))
        self.assertEqual(model.sar_stem.downsample.stride, (2, 2))
        self.assertEqual(model.sar_stem.downsample.padding, (0, 0))
        self.assertEqual(model.hsi_stem.downsample.kernel_size, (2, 2))
        self.assertEqual(model.hsi_stem.downsample.stride, (2, 2))
        self.assertEqual(model.hsi_stem.downsample.padding, (0, 0))
        self.assertFalse(hasattr(model.sar_stem, "down1"))
        self.assertFalse(hasattr(model.sar_stem, "down2"))
        self.assertFalse(hasattr(model.hsi_stem, "down1"))
        self.assertFalse(hasattr(model.hsi_stem, "down2"))
        self.assertEqual(len(model.wrapper_blocks), 1)
        self.assertIsInstance(wrapper, LCRWrapperBlock)
        self.assertEqual(len(wrapper.local_blocks), 1)
        self.assertEqual(len(wrapper.global_blocks), 1)
        self.assertIsInstance(wrapper.local_blocks[0].attn, NeighborhoodCrossAttention)
        self.assertIsInstance(wrapper.global_blocks[0], GlobalBlock)
        self.assertIsInstance(wrapper.global_blocks[0].attn, GlobalSelfAttention)
        self.assertIsInstance(model.candidate_decoder, LatentDecoder)
        self.assertIsInstance(model.mask_decoder, LatentDecoder)
        self.assertIsNot(model.candidate_decoder, model.mask_decoder)
        self.assertEqual(model.candidate_decoder.full_dim, 16)
        self.assertIsInstance(model.candidate_decoder.project, torch.nn.Conv2d)
        self.assertEqual(model.candidate_decoder.project.kernel_size, (1, 1))
        self.assertEqual(model.candidate_decoder.project.in_channels, 32)
        self.assertEqual(model.candidate_decoder.project.out_channels, 16)
        self.assertIsInstance(model.candidate_decoder.upsample, BilinearUp2x)
        self.assertEqual(model.candidate_decoder.out.kernel_size, (1, 1))
        self.assertEqual(model.candidate_head[0].in_channels, 16)
        self.assertEqual(model.candidate_head[0].out_channels, 16)
        self.assertEqual(model.candidate_head[0].kernel_size, (1, 1))
        self.assertEqual(model.candidate_head[2].kernel_size, (1, 1))
        self.assertEqual(model.candidate_head[2].in_channels, 16)
        self.assertIsInstance(model.mask_decoder.project, torch.nn.Conv2d)
        self.assertEqual(model.mask_decoder.project.out_channels, 16)
        self.assertIsInstance(model.mask_decoder.upsample, BilinearUp2x)
        self.assertEqual(model.mask_out.kernel_size, (1, 1))
        self.assertEqual(model.mask_out.in_channels, 16)
        self.assertFalse(hasattr(model, "local_blocks"))
        self.assertFalse(hasattr(model, "global_blocks"))
        self.assertFalse(hasattr(model, "reconstruction"))
        self.assertFalse(hasattr(model, "mask_seed"))
        self.assertFalse(hasattr(model, "mask_merge_h2"))
        self.assertFalse(hasattr(model.candidate_decoder, "h2_dim"))
        self.assertFalse(hasattr(model.candidate_decoder, "upsample_h2"))
        self.assertFalse(hasattr(model.candidate_decoder, "upsample_full"))
        self.assertFalse(any(isinstance(module, torch.nn.PixelShuffle) for module in model.modules()))

    def test_wrapper_block_tracks_configured_inner_block_counts(self) -> None:
        model = LCR(
            dim=32,
            num_blocks=2,
            local_block_count=3,
            global_block_count=2,
            heads=4,
            neighborhood_size=7,
        )

        self.assertEqual(model.num_blocks, 2)
        self.assertEqual(model.local_block_count, 3)
        self.assertEqual(model.global_block_count, 2)
        self.assertEqual(len(model.wrapper_blocks), 2)
        self.assertTrue(all(isinstance(wrapper, LCRWrapperBlock) for wrapper in model.wrapper_blocks))
        self.assertTrue(all(len(wrapper.local_blocks) == 3 for wrapper in model.wrapper_blocks))
        self.assertTrue(all(len(wrapper.global_blocks) == 2 for wrapper in model.wrapper_blocks))

    def test_lcr_contains_no_3x3_convolutions(self) -> None:
        model = LCR(dim=32, num_blocks=1, heads=4, neighborhood_size=7)

        kernel_sizes = [
            module.kernel_size
            for module in model.modules()
            if isinstance(module, torch.nn.Conv2d)
        ]
        self.assertTrue(kernel_sizes)
        self.assertFalse(any(kernel_size == (3, 3) for kernel_size in kernel_sizes))

    def test_global_block_uses_learned_downsampling_and_bilinear_restore(self) -> None:
        block = GlobalBlock(dim=32, heads=4, ffn_expansion=2)
        upsample_source = inspect.getsource(BilinearUp2x.forward)

        self.assertIn("interpolate", upsample_source)
        self.assertIn("bilinear", upsample_source)
        self.assertIn("align_corners=False", upsample_source)
        self.assertIsInstance(block.z_downsample, torch.nn.Conv2d)
        self.assertEqual(block.z_downsample.kernel_size, (2, 2))
        self.assertEqual(block.z_downsample.stride, (2, 2))
        self.assertFalse(hasattr(block, "h_downsample"))
        self.assertFalse(hasattr(block, "h_attn_norm"))
        self.assertIsInstance(block.upsample, BilinearUp2x)

    def test_global_block_accepts_z_only_input(self) -> None:
        block = GlobalBlock(dim=32, heads=4, ffn_expansion=2)
        z = torch.randn(1, 32, 8, 8)

        out = block(z)

        self.assertEqual(out.shape, z.shape)

    def test_bilinear_upsample_doubles_spatial_size_and_preserves_channels(self) -> None:
        upsample = BilinearUp2x()
        x = torch.randn(1, 4, 2, 3)

        y = upsample(x)
        z = upsample(x, size=(5, 7))

        self.assertEqual(y.shape, (1, 4, 4, 6))
        self.assertEqual(z.shape, (1, 4, 5, 7))

    def test_lcr_model_source_uses_bilinear_interpolate(self) -> None:
        source = inspect.getsource(lcr_model)

        self.assertIn("interpolate", source)
        self.assertIn("mode=\"bilinear\"", source)
        self.assertIn("align_corners=False", source)
        self.assertNotIn("adaptive_avg_pool2d", source)
        self.assertNotIn("max_pool2d", source)

    def test_mask_out_bias_uses_mask_bias_init(self) -> None:
        model = LCR(dim=32, num_blocks=1, heads=4, neighborhood_size=7, mask_bias_init=-1.5)

        self.assertTrue(torch.allclose(model.mask_out.bias, torch.full_like(model.mask_out.bias, -1.5)))

    def test_even_neighborhood_size_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive odd integer"):
            LCR(dim=32, num_blocks=1, heads=4, neighborhood_size=8)

    def test_non_positive_neighborhood_size_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive odd integer"):
            LCR(dim=32, num_blocks=1, heads=4, neighborhood_size=0)

    def test_non_positive_local_block_count_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "local_block_count must be greater than zero"):
            LCR(dim=32, num_blocks=1, local_block_count=0, heads=4, neighborhood_size=7)

    def test_non_positive_global_block_count_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "global_block_count must be greater than zero"):
            LCR(dim=32, num_blocks=1, global_block_count=0, heads=4, neighborhood_size=7)

    def test_window_size_keyword_is_no_longer_supported(self) -> None:
        with self.assertRaises(TypeError):
            LCR(dim=32, num_blocks=1, heads=4, window_size=7)


if __name__ == "__main__":
    unittest.main()
