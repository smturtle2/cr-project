from __future__ import annotations

import unittest

import torch

from modules.loss_fn import LCRLoss
from modules.model.lcr import LCR
from modules.model.lcr.model import (
    Attn,
    AttnBlock,
    LCRWrapperBlock,
    LatentDecoder,
    LatentEncoder,
    _AttnCore,
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
            cross_block_count=2,
            self_block_count=3,
            heads=4,
        )

        sar = torch.randn(2, 2, 64, 64)
        cloudy = torch.randn(2, 13, 64, 64)

        prediction = model(sar, cloudy)

        self.assertEqual(prediction.shape, (2, 13, 64, 64))

    def test_forward_handles_spatial_size_divisible_by_two_but_not_four(self) -> None:
        model = LCR(dim=32, num_blocks=1, heads=4)

        sar = torch.randn(1, 2, 34, 38)
        cloudy = torch.randn(1, 13, 34, 38)

        prediction = model(sar, cloudy)

        self.assertEqual(prediction.shape, (1, 13, 34, 38))

    def test_forward_rejects_odd_spatial_size(self) -> None:
        model = LCR(dim=32, num_blocks=1, heads=4)

        sar = torch.randn(1, 2, 246, 261)
        cloudy = torch.randn(1, 13, 246, 261)

        with self.assertRaisesRegex(ValueError, "divisible by 2"):
            model(sar, cloudy)

    def test_mismatched_spatial_size_raises(self) -> None:
        model = LCR(dim=32, num_blocks=1, heads=4)

        sar = torch.randn(1, 2, 256, 256)
        cloudy = torch.randn(1, 13, 255, 256)

        with self.assertRaisesRegex(ValueError, "same batch and spatial size"):
            model(sar, cloudy)

    def test_lcr_loss_backward_is_finite(self) -> None:
        model = LCR(
            dim=32,
            num_blocks=1,
            cross_block_count=2,
            self_block_count=2,
            heads=4,
        )
        loss_fn = LCRLoss()

        sar = torch.randn(1, 2, 32, 32)
        cloudy = torch.randn(1, 13, 32, 32)
        target = torch.randn(1, 13, 32, 32)

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

    def test_cross_and_self_attn_parameters_receive_gradients(self) -> None:
        model = LCR(
            dim=32,
            num_blocks=1,
            cross_block_count=1,
            self_block_count=1,
            heads=4,
        )
        loss_fn = LCRLoss()

        sar = torch.randn(1, 2, 32, 32)
        cloudy = torch.randn(1, 13, 32, 32)
        target = torch.randn(1, 13, 32, 32)

        prediction = model(sar, cloudy)
        loss = loss_fn(prediction, target)
        loss.backward()

        for wrapper_block in model.wrapper_blocks:
            cross_attn = wrapper_block.cross_blocks[0].attn
            self_attn = wrapper_block.self_blocks[0].attn
            for attn in (cross_attn, self_attn):
                grad_tensors = [
                    attn.q_proj.weight.grad,
                    attn.k_proj.weight.grad,
                    attn.v_proj.weight.grad,
                    attn.out_proj.weight.grad,
                ]
                self.assertTrue(
                    all(
                        grad is not None
                        and bool(torch.isfinite(grad).all().item())
                        and float(grad.abs().sum()) > 0.0
                        for grad in grad_tensors
                    )
                )

    def test_wrapper_blocks_use_distinct_parameters_across_depth(self) -> None:
        model = LCR(dim=32, num_blocks=2, heads=4)

        self.assertEqual(len(model.wrapper_blocks), 2)
        self.assertIsNot(model.wrapper_blocks[0], model.wrapper_blocks[1])
        self.assertIsNot(
            model.wrapper_blocks[0].cross_blocks[0].attn.q_proj.weight,
            model.wrapper_blocks[1].cross_blocks[0].attn.q_proj.weight,
        )
        self.assertIsNot(
            model.wrapper_blocks[0].self_blocks[0].attn.q_proj.weight,
            model.wrapper_blocks[1].self_blocks[0].attn.q_proj.weight,
        )

    def test_exclude_self_value_component_removes_self_projection(self) -> None:
        attn_out = torch.tensor([[[3.0, 4.0], [1.0, 2.0]]], dtype=torch.float32)
        self_value = torch.tensor([[[0.0, 5.0], [2.0, 0.0]]], dtype=torch.float32)

        out = _exclude_self_value_component(attn_out, self_value)
        self_value_unit = torch.nn.functional.normalize(self_value, dim=-1)
        alignment = (out * self_value_unit).sum(dim=-1)

        self.assertTrue(torch.allclose(alignment, torch.zeros_like(alignment), atol=1e-6))

    def test_attn_core_supports_dense_context(self) -> None:
        core = _AttnCore()

        dense_query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float32)
        dense_key = dense_query.clone()
        dense_value = dense_query.clone()

        dense_out = core(dense_query, dense_key, dense_value)

        self.assertEqual(dense_out.shape, dense_query.shape)
        self.assertTrue(bool(torch.isfinite(dense_out).all().item()))

    def test_attn_core_dense_path_uses_sdpa_default_scale(self) -> None:
        core = _AttnCore()
        query = torch.tensor(
            [[[[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]]]],
            dtype=torch.float32,
        )
        key = torch.tensor(
            [[[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]]]],
            dtype=torch.float32,
        )
        value = torch.tensor(
            [[[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]]],
            dtype=torch.float32,
        )
        out = core(query, key, value)
        scores = query @ key.transpose(-2, -1) * (query.shape[-1] ** -0.5)
        expected = scores.softmax(dim=-1) @ value

        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for flash attention smoke test")
    def test_attn_core_dense_cuda_fp32_uses_flash_compatible_bf16_inputs(self) -> None:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        core = _AttnCore().cuda()
        query = torch.randn(1, 6, 128, 40, device="cuda", dtype=torch.float32)
        key = torch.randn(1, 6, 128, 40, device="cuda", dtype=torch.float32)
        value = torch.randn(1, 6, 128, 40, device="cuda", dtype=torch.float32)

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = core(query, key, value)
        torch.cuda.synchronize()

        self.assertEqual(out.shape, query.shape)
        self.assertEqual(out.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(out).all().item()))

    def test_attn_uses_attn_core_for_cross_and_self_contexts(self) -> None:
        attn = Attn(dim=4, heads=2)
        query = torch.randn(1, 4, 3, 5)
        context = torch.randn(1, 4, 3, 5)

        cross_out = attn(query, context)
        self_out = attn(query)

        self.assertIsInstance(attn.core, _AttnCore)
        self.assertEqual(cross_out.shape, query.shape)
        self.assertEqual(self_out.shape, query.shape)
        self.assertTrue(bool(torch.isfinite(cross_out).all().item()))
        self.assertTrue(bool(torch.isfinite(self_out).all().item()))

    def test_attn_excludes_self_value_direction(self) -> None:
        attn = Attn(dim=4, heads=2)
        for proj in (attn.q_proj, attn.k_proj, attn.v_proj, attn.out_proj):
            self._set_identity_projection(proj)

        x = torch.zeros(1, 4, 2, 2)
        x[:, 0] = 1.0
        x[:, 2] = 1.0

        out = attn(x, exclude_self_value=True)

        self.assertTrue(torch.allclose(out, torch.zeros_like(out), atol=1e-6))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for flash attention smoke test")
    def test_attn_self_cuda_fp32_uses_flash_compatible_bf16_inputs(self) -> None:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        attn = Attn(dim=240, heads=6).cuda().eval()
        x = torch.randn(1, 240, 16, 16, device="cuda", dtype=torch.float32)

        with torch.no_grad(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = attn(x, exclude_self_value=True)
        torch.cuda.synchronize()

        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(out).all().item()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for flash attention smoke test")
    def test_attn_cross_cuda_fp32_uses_flash_compatible_bf16_inputs(self) -> None:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        attn = Attn(dim=240, heads=6).cuda().eval()
        query = torch.randn(1, 240, 16, 16, device="cuda", dtype=torch.float32)
        context = torch.randn(1, 240, 16, 16, device="cuda", dtype=torch.float32)

        with torch.no_grad(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = attn(query, context)
        torch.cuda.synchronize()

        self.assertEqual(out.shape, query.shape)
        self.assertEqual(out.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(out).all().item()))

    def test_attn_cross_uses_full_context(self) -> None:
        attn = Attn(dim=4, heads=2)
        query = torch.randn(1, 4, 3, 5)
        context = torch.randn(1, 4, 3, 5)

        out = attn(query, context)

        self.assertEqual(out.shape, query.shape)
        self.assertTrue(bool(torch.isfinite(out).all().item()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for flash attention smoke test")
    def test_lcr_cuda_fp32_uses_flash_attention_backend(self) -> None:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        model = LCR(dim=32, num_blocks=1, cross_block_count=1, self_block_count=1, heads=4).cuda().eval()
        sar = torch.randn(1, 2, 32, 32, device="cuda", dtype=torch.float32)
        cloudy = torch.randn(1, 13, 32, 32, device="cuda", dtype=torch.float32)

        with torch.no_grad(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            prediction = model(sar, cloudy)
        torch.cuda.synchronize()

        self.assertEqual(prediction.shape, cloudy.shape)
        self.assertEqual(prediction.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(prediction).all().item()))

    def test_model_uses_attn_block_encoders_and_pixelshuffle_decoder(self) -> None:
        model = LCR(
            dim=32,
            num_blocks=1,
            cross_block_count=1,
            self_block_count=1,
            heads=4,
        )
        wrapper = model.wrapper_blocks[0]

        self.assertIsInstance(model.sar_encoder, LatentEncoder)
        self.assertIsInstance(model.hsi_encoder, LatentEncoder)
        self.assertIsInstance(model.sar_encoder.downsample, torch.nn.Conv2d)
        self.assertIsInstance(model.hsi_encoder.downsample, torch.nn.Conv2d)
        self.assertEqual(model.sar_encoder.downsample.kernel_size, (2, 2))
        self.assertEqual(model.sar_encoder.downsample.stride, (2, 2))
        self.assertEqual(model.sar_encoder.downsample.padding, (0, 0))
        self.assertEqual(model.hsi_encoder.downsample.kernel_size, (2, 2))
        self.assertEqual(model.hsi_encoder.downsample.stride, (2, 2))
        self.assertEqual(model.hsi_encoder.downsample.padding, (0, 0))
        self.assertEqual(len(model.sar_encoder.self_blocks), 1)
        self.assertEqual(len(model.hsi_encoder.self_blocks), 1)
        self.assertIsInstance(model.sar_encoder.self_blocks[0], AttnBlock)
        self.assertIsInstance(model.sar_encoder.self_blocks[0].attn, Attn)
        self.assertIsInstance(model.hsi_encoder.self_blocks[0], AttnBlock)
        self.assertIsInstance(model.hsi_encoder.self_blocks[0].attn, Attn)
        self.assertEqual(len(model.wrapper_blocks), 1)
        self.assertIsInstance(model.wrapper_blocks, torch.nn.ModuleList)
        self.assertIsInstance(wrapper, LCRWrapperBlock)
        self.assertEqual(len(wrapper.cross_blocks), 1)
        self.assertEqual(len(wrapper.self_blocks), 1)
        self.assertIsInstance(wrapper.cross_blocks[0], AttnBlock)
        self.assertIsInstance(wrapper.cross_blocks[0].attn, Attn)
        self.assertIsInstance(wrapper.self_blocks[0], AttnBlock)
        self.assertIsInstance(wrapper.self_blocks[0].attn, Attn)
        self.assertIsInstance(model.candidate_decoder, LatentDecoder)
        self.assertIsInstance(model.mask_decoder, LatentDecoder)
        self.assertIsNot(model.candidate_decoder, model.mask_decoder)
        self.assertEqual(model.candidate_decoder.full_dim, 16)
        self.assertIsInstance(model.candidate_decoder.expand, torch.nn.Conv2d)
        self.assertEqual(model.candidate_decoder.expand.kernel_size, (1, 1))
        self.assertEqual(model.candidate_decoder.expand.in_channels, 32)
        self.assertEqual(model.candidate_decoder.expand.out_channels, 64)
        self.assertIsInstance(model.candidate_decoder.upsample, torch.nn.PixelShuffle)
        self.assertEqual(model.candidate_decoder.upsample.upscale_factor, 2)
        self.assertEqual(model.candidate_decoder.out.kernel_size, (1, 1))
        self.assertEqual(model.candidate_head[0].in_channels, 16)
        self.assertEqual(model.candidate_head[0].out_channels, 16)
        self.assertEqual(model.candidate_head[0].kernel_size, (1, 1))
        self.assertEqual(model.candidate_head[2].kernel_size, (1, 1))
        self.assertEqual(model.candidate_head[2].in_channels, 16)
        self.assertIsInstance(model.mask_decoder.expand, torch.nn.Conv2d)
        self.assertEqual(model.mask_decoder.expand.out_channels, 64)
        self.assertIsInstance(model.mask_decoder.upsample, torch.nn.PixelShuffle)
        self.assertEqual(model.mask_decoder.upsample.upscale_factor, 2)
        self.assertEqual(model.mask_out.kernel_size, (1, 1))
        self.assertEqual(model.mask_out.in_channels, 16)
        self.assertTrue(hasattr(model, "wrapper_blocks"))
        self.assertTrue(any(isinstance(module, torch.nn.PixelShuffle) for module in model.modules()))

    def test_wrapper_stack_tracks_configured_inner_block_counts(self) -> None:
        model = LCR(
            dim=32,
            num_blocks=2,
            cross_block_count=3,
            self_block_count=2,
            heads=4,
        )

        self.assertEqual(model.num_blocks, 2)
        self.assertEqual(model.cross_block_count, 3)
        self.assertEqual(model.self_block_count, 2)
        self.assertEqual(len(model.wrapper_blocks), 2)
        self.assertTrue(all(isinstance(wrapper, LCRWrapperBlock) for wrapper in model.wrapper_blocks))
        self.assertTrue(all(len(wrapper.cross_blocks) == 3 for wrapper in model.wrapper_blocks))
        self.assertTrue(all(len(wrapper.self_blocks) == 2 for wrapper in model.wrapper_blocks))
        self.assertEqual(len(model.sar_encoder.self_blocks), 2)
        self.assertEqual(len(model.hsi_encoder.self_blocks), 2)

    def test_lcr_contains_no_3x3_convolutions(self) -> None:
        model = LCR(dim=32, num_blocks=1, heads=4)

        kernel_sizes = [
            module.kernel_size
            for module in model.modules()
            if isinstance(module, torch.nn.Conv2d)
        ]
        self.assertTrue(kernel_sizes)
        self.assertFalse(any(kernel_size == (3, 3) for kernel_size in kernel_sizes))

    def test_attn_block_uses_same_resolution_attn(self) -> None:
        block = AttnBlock(dim=32, heads=4, ffn_expansion=2)
        self.assertIsInstance(block.attn, Attn)

    def test_attn_block_accepts_z_only_input(self) -> None:
        block = AttnBlock(dim=32, heads=4, ffn_expansion=2)
        z = torch.randn(1, 32, 8, 8)

        out = block(z)

        self.assertEqual(out.shape, z.shape)

    def test_latent_decoder_pixelshuffle_doubles_spatial_size_and_sets_full_dim_channels(self) -> None:
        decoder = LatentDecoder(dim=8, ffn_expansion=2)
        latent = torch.randn(1, 8, 5, 7)

        decoded = decoder(latent)

        self.assertEqual(decoder.full_dim, 4)
        self.assertEqual(decoded.shape, (1, 4, 10, 14))

    def test_mask_out_bias_uses_mask_bias_init(self) -> None:
        model = LCR(dim=32, num_blocks=1, heads=4, mask_bias_init=-1.5)

        self.assertTrue(torch.allclose(model.mask_out.bias, torch.full_like(model.mask_out.bias, -1.5)))


if __name__ == "__main__":
    unittest.main()
