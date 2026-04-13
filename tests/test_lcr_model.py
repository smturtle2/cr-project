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
    GlobalSelfAttn,
    LCRWrapperBlock,
    LatentDecoder,
    NeighborhoodAttn,
    _AttentionCore,
    _exclude_self_value_component,
)


class LCRModelTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    @staticmethod
    def _set_identity_projection(conv: torch.nn.Conv2d) -> None:
        with torch.no_grad():
            torch.nn.init.dirac_(conv.weight)
            if conv.bias is not None:
                conv.bias.zero_()

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

    def test_global_self_attention_parameters_receive_gradients_in_both_branches(self) -> None:
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

        for global_attn in (
            model.c_wrapper_blocks[0].global_blocks[0].attn,
            model.m_wrapper_blocks[0].global_blocks[0].attn,
        ):
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

    def test_c_and_m_branches_use_distinct_wrapper_parameters(self) -> None:
        model = LCR(dim=32, num_blocks=2, heads=4, neighborhood_size=7)

        self.assertEqual(len(model.c_wrapper_blocks), 2)
        self.assertEqual(len(model.m_wrapper_blocks), 2)
        self.assertIsNot(model.c_wrapper_blocks[0], model.m_wrapper_blocks[0])
        self.assertIsNot(
            model.c_wrapper_blocks[0].global_blocks[0].attn.q_proj.weight,
            model.m_wrapper_blocks[0].global_blocks[0].attn.q_proj.weight,
        )

    def test_exclude_self_value_component_removes_self_projection(self) -> None:
        attn_out = torch.tensor([[[3.0, 4.0], [1.0, 2.0]]], dtype=torch.float32)
        self_value = torch.tensor([[[0.0, 5.0], [2.0, 0.0]]], dtype=torch.float32)

        out = _exclude_self_value_component(attn_out, self_value)
        self_value_unit = torch.nn.functional.normalize(self_value, dim=-1)
        alignment = (out * self_value_unit).sum(dim=-1)

        self.assertTrue(torch.allclose(alignment, torch.zeros_like(alignment), atol=1e-6))

    def test_attention_core_supports_dense_and_gathered_contexts(self) -> None:
        core = _AttentionCore(scale=1.0)

        dense_query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float32)
        dense_key = dense_query.clone()
        dense_value = dense_query.clone()
        dense_mask = torch.tensor([[[[True, False], [True, True]]]])

        dense_out = core(dense_query, dense_key, dense_value, valid_mask=dense_mask)

        sparse_query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float32)
        sparse_key = torch.tensor(
            [[[[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]]],
            dtype=torch.float32,
        )
        sparse_value = sparse_key.clone()
        sparse_mask = torch.tensor([[[[True, False], [True, True]]]])

        sparse_out = core(
            sparse_query,
            sparse_key,
            sparse_value,
            valid_mask=sparse_mask,
            self_value=sparse_query,
        )

        self.assertEqual(dense_out.shape, dense_query.shape)
        self.assertEqual(sparse_out.shape, sparse_query.shape)
        self.assertTrue(bool(torch.isfinite(dense_out).all().item()))
        self.assertTrue(bool(torch.isfinite(sparse_out).all().item()))

    def test_neighborhood_and_global_attn_share_attention_core(self) -> None:
        neighborhood_attn = NeighborhoodAttn(dim=4, heads=2, neighborhood_size=3)
        global_attn = GlobalSelfAttn(dim=4, heads=2)

        self.assertIsInstance(neighborhood_attn.core, _AttentionCore)
        self.assertIsInstance(global_attn.core, _AttentionCore)

    def test_global_self_attn_excludes_self_value_direction(self) -> None:
        attn = GlobalSelfAttn(dim=4, heads=2)
        for proj in (attn.q_proj, attn.k_proj, attn.v_proj, attn.out_proj):
            self._set_identity_projection(proj)

        x = torch.zeros(1, 4, 2, 2)
        x[:, 0] = 1.0
        x[:, 2] = 1.0

        out = attn(x)

        self.assertTrue(torch.allclose(out, torch.zeros_like(out), atol=1e-6))

    def test_neighborhood_attn_applies_xsa_only_in_query_only_mode(self) -> None:
        attn = NeighborhoodAttn(dim=2, heads=1, neighborhood_size=3)
        for proj in (attn.q_proj, attn.k_proj, attn.v_proj, attn.out_proj):
            self._set_identity_projection(proj)

        x = torch.zeros(1, 2, 2, 2)
        x[:, 0] = 1.0

        self_mode_out = attn(x)
        explicit_kv_out = attn(x, x, x)

        self_alignment = (self_mode_out[:, 0] * x[:, 0] + self_mode_out[:, 1] * x[:, 1]).abs().max()
        explicit_alignment = (explicit_kv_out[:, 0] * x[:, 0] + explicit_kv_out[:, 1] * x[:, 1]).abs().max()

        self.assertTrue(torch.allclose(self_mode_out, torch.zeros_like(self_mode_out), atol=1e-6))
        self.assertGreater(float(explicit_alignment.detach()), 0.5)
        self.assertLess(float(self_alignment.detach()), 1e-6)

    def test_model_uses_single_downsample_stem_and_bilinear_decoder(self) -> None:
        model = LCR(
            dim=32,
            num_blocks=1,
            local_block_count=1,
            global_block_count=1,
            heads=4,
            neighborhood_size=7,
        )
        c_wrapper = model.c_wrapper_blocks[0]
        m_wrapper = model.m_wrapper_blocks[0]

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
        self.assertEqual(len(model.c_wrapper_blocks), 1)
        self.assertEqual(len(model.m_wrapper_blocks), 1)
        self.assertIsInstance(c_wrapper, LCRWrapperBlock)
        self.assertIsInstance(m_wrapper, LCRWrapperBlock)
        self.assertIsNot(c_wrapper, m_wrapper)
        self.assertEqual(len(c_wrapper.local_blocks), 1)
        self.assertEqual(len(c_wrapper.global_blocks), 1)
        self.assertEqual(len(m_wrapper.local_blocks), 1)
        self.assertEqual(len(m_wrapper.global_blocks), 1)
        self.assertIsInstance(c_wrapper.local_blocks[0].attn, NeighborhoodAttn)
        self.assertIsInstance(c_wrapper.global_blocks[0], GlobalBlock)
        self.assertIsInstance(c_wrapper.global_blocks[0].attn, GlobalSelfAttn)
        self.assertIsInstance(m_wrapper.local_blocks[0].attn, NeighborhoodAttn)
        self.assertIsInstance(m_wrapper.global_blocks[0], GlobalBlock)
        self.assertIsInstance(m_wrapper.global_blocks[0].attn, GlobalSelfAttn)
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
        self.assertFalse(hasattr(model, "wrapper_blocks"))
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
        self.assertEqual(len(model.c_wrapper_blocks), 2)
        self.assertEqual(len(model.m_wrapper_blocks), 2)
        self.assertTrue(all(isinstance(wrapper, LCRWrapperBlock) for wrapper in model.c_wrapper_blocks))
        self.assertTrue(all(isinstance(wrapper, LCRWrapperBlock) for wrapper in model.m_wrapper_blocks))
        self.assertTrue(all(len(wrapper.local_blocks) == 3 for wrapper in model.c_wrapper_blocks))
        self.assertTrue(all(len(wrapper.global_blocks) == 2 for wrapper in model.c_wrapper_blocks))
        self.assertTrue(all(len(wrapper.local_blocks) == 3 for wrapper in model.m_wrapper_blocks))
        self.assertTrue(all(len(wrapper.global_blocks) == 2 for wrapper in model.m_wrapper_blocks))

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
