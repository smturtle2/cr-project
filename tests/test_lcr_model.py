from __future__ import annotations

import unittest

import torch

from modules.loss_fn import LCRLoss
from modules.model.lcr import LCR
from modules.model.lcr.model import (
    Attn,
    AttnBlock,
    EncoderSpatialBlock,
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

    def test_forward_uses_configured_patch_size(self) -> None:
        model = LCR(dim=32, num_blocks=1, heads=4, patch_size=4)

        sar = torch.randn(1, 2, 16, 20)
        cloudy = torch.randn(1, 13, 16, 20)

        prediction = model(sar, cloudy)

        self.assertEqual(model.patch_size, 4)
        self.assertEqual(prediction.shape, (1, 13, 16, 20))

    def test_default_initialization_does_not_collapse_to_cloudy_input(self) -> None:
        model = LCR(dim=32, num_blocks=1, heads=4).eval()

        sar = torch.randn(1, 2, 32, 32)
        cloudy = torch.randn(1, 13, 32, 32)

        with torch.no_grad():
            prediction = model(sar, cloudy)

        self.assertGreater(float((prediction - cloudy).abs().mean()), 1e-3)

    def test_default_initialization_keeps_mask_and_trunk_gradients_alive(self) -> None:
        model = LCR(dim=32, num_blocks=1, cross_block_count=1, self_block_count=1, heads=4)
        loss_fn = LCRLoss()

        sar = torch.randn(1, 2, 32, 32)
        cloudy = torch.randn(1, 13, 32, 32)
        target = torch.randn(1, 13, 32, 32)

        loss = loss_fn(model(sar, cloudy), target)
        loss.backward()

        mask_bias_grad = model.mask_out.bias.grad
        cross_q_grad = model.wrapper_blocks[0].cross_blocks[0].attn.q_proj.weight.grad
        encoder_conv_grad = model.sar_encoder.encoder_blocks[0].depthwise.weight.grad

        self.assertIsNotNone(mask_bias_grad)
        self.assertIsNotNone(cross_q_grad)
        self.assertIsNotNone(encoder_conv_grad)
        self.assertGreater(float(mask_bias_grad.abs().mean()), 1e-4)
        self.assertGreater(float(cross_q_grad.abs().mean()), 1e-7)
        self.assertGreater(float(encoder_conv_grad.abs().mean()), 1e-7)

    def test_mismatched_spatial_size_raises(self) -> None:
        model = LCR(dim=32, num_blocks=1, heads=4)

        sar = torch.randn(1, 2, 256, 256)
        cloudy = torch.randn(1, 13, 255, 256)

        with self.assertRaisesRegex(ValueError, "same batch and spatial size"):
            model(sar, cloudy)

    def test_invalid_patch_size_config_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "patch_size must be greater than zero"):
            LCR(dim=32, num_blocks=1, heads=4, patch_size=0)

    def test_invalid_encoder_block_count_config_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "encoder_block_count must be greater than zero"):
            LCR(dim=32, num_blocks=1, heads=4, encoder_block_count=0)

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

    def test_latent_encoder_uses_patch_embedding_grid_and_conv_residual_blocks(self) -> None:
        encoder = LatentEncoder(
            in_channels=2,
            dim=8,
            ffn_expansion=2,
            encoder_block_count=1,
            patch_size=4,
        )
        x = torch.randn(1, 2, 16, 20)

        encoded = encoder(x)

        self.assertEqual(encoder.patch_size, 4)
        self.assertIsInstance(encoder.proj, torch.nn.Conv2d)
        self.assertEqual(encoder.proj.kernel_size, (1, 1))
        self.assertEqual(encoder.proj.in_channels, 2)
        self.assertEqual(encoder.proj.out_channels, 8)
        self.assertIsInstance(encoder.patch_embed, torch.nn.Conv2d)
        self.assertEqual(encoder.patch_embed.kernel_size, (4, 4))
        self.assertEqual(encoder.patch_embed.stride, (4, 4))
        self.assertEqual(encoder.patch_embed.in_channels, 8)
        self.assertEqual(encoder.patch_embed.out_channels, 8)
        self.assertFalse(hasattr(encoder, "self_blocks"))
        self.assertEqual(len(encoder.encoder_blocks), 1)
        self.assertIsInstance(encoder.encoder_blocks[0], EncoderSpatialBlock)
        self.assertEqual(encoder.encoder_blocks[0].depthwise.kernel_size, (3, 3))
        self.assertEqual(encoder.encoder_blocks[0].depthwise.padding, (1, 1))
        self.assertEqual(encoder.encoder_blocks[0].depthwise.groups, 8)
        self.assertFalse(any(isinstance(module, AttnBlock) for module in encoder.modules()))
        self.assertEqual(encoded.shape, (1, 8, 4, 5))

    def test_lcr_attention_receives_patch_grid_tokens(self) -> None:
        model = LCR(
            dim=8,
            num_blocks=1,
            cross_block_count=1,
            self_block_count=1,
            heads=2,
            patch_size=4,
        )
        seen: dict[str, tuple[int, ...]] = {}

        class RecordingCore(torch.nn.Module):
            def forward(
                self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                *,
                self_value: torch.Tensor | None = None,
            ) -> torch.Tensor:
                seen["query_shape"] = tuple(query.shape)
                seen["key_shape"] = tuple(key.shape)
                seen["value_shape"] = tuple(value.shape)
                return value

        model.wrapper_blocks[0].cross_blocks[0].attn.core = RecordingCore()
        sar = torch.randn(1, 2, 16, 20)
        cloudy = torch.randn(1, 13, 16, 20)

        prediction = model(sar, cloudy)

        self.assertEqual(prediction.shape, cloudy.shape)
        self.assertEqual(seen["query_shape"], (1, 2, 20, 4))
        self.assertEqual(seen["key_shape"], (1, 2, 20, 4))
        self.assertEqual(seen["value_shape"], (1, 2, 20, 4))

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

    def test_model_uses_patch_encoders_candidate_blend_and_pixelshuffle_decoder(self) -> None:
        model = LCR(
            dim=32,
            num_blocks=1,
            cross_block_count=1,
            self_block_count=1,
            heads=4,
            patch_size=2,
        )
        wrapper = model.wrapper_blocks[0]

        self.assertIsInstance(model.sar_encoder, LatentEncoder)
        self.assertIsInstance(model.hsi_encoder, LatentEncoder)
        self.assertEqual(model.sar_encoder.patch_size, 2)
        self.assertIsInstance(model.sar_encoder.proj, torch.nn.Conv2d)
        self.assertEqual(model.sar_encoder.proj.kernel_size, (1, 1))
        self.assertEqual(model.sar_encoder.proj.in_channels, 2)
        self.assertEqual(model.hsi_encoder.proj.in_channels, 13)
        self.assertEqual(model.sar_encoder.proj.out_channels, 32)
        self.assertEqual(model.hsi_encoder.proj.out_channels, 32)
        self.assertIsInstance(model.sar_encoder.patch_embed, torch.nn.Conv2d)
        self.assertEqual(model.sar_encoder.patch_embed.kernel_size, (2, 2))
        self.assertEqual(model.sar_encoder.patch_embed.stride, (2, 2))
        self.assertFalse(hasattr(model.sar_encoder, "self_blocks"))
        self.assertFalse(hasattr(model.hsi_encoder, "self_blocks"))
        self.assertEqual(len(model.sar_encoder.encoder_blocks), 1)
        self.assertEqual(len(model.hsi_encoder.encoder_blocks), 1)
        self.assertIsInstance(model.sar_encoder.encoder_blocks[0], EncoderSpatialBlock)
        self.assertIsInstance(model.hsi_encoder.encoder_blocks[0], EncoderSpatialBlock)
        self.assertEqual(model.sar_encoder.encoder_blocks[0].depthwise.kernel_size, (3, 3))
        self.assertEqual(model.sar_encoder.encoder_blocks[0].depthwise.groups, 32)
        self.assertEqual(model.hsi_encoder.encoder_blocks[0].depthwise.groups, 32)
        self.assertFalse(any(isinstance(module, AttnBlock) for module in model.sar_encoder.modules()))
        self.assertFalse(any(isinstance(module, AttnBlock) for module in model.hsi_encoder.modules()))
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
        self.assertFalse(hasattr(model, "delta_decoder"))
        self.assertFalse(hasattr(model, "delta_head"))
        self.assertEqual(model.candidate_decoder.full_dim, 16)
        self.assertIsInstance(model.candidate_decoder.expand, torch.nn.Conv2d)
        self.assertEqual(model.candidate_decoder.expand.kernel_size, (1, 1))
        self.assertEqual(model.candidate_decoder.expand.in_channels, 32)
        self.assertEqual(model.candidate_decoder.expand.out_channels, 16 * 2 * 2)
        self.assertIsInstance(model.candidate_decoder.upsample, torch.nn.PixelShuffle)
        self.assertEqual(model.candidate_decoder.upsample.upscale_factor, 2)
        self.assertEqual(model.candidate_decoder.out.kernel_size, (1, 1))
        self.assertEqual(model.candidate_head[0].in_channels, 16)
        self.assertEqual(model.candidate_head[0].out_channels, 16)
        self.assertEqual(model.candidate_head[0].kernel_size, (1, 1))
        self.assertEqual(model.candidate_head[2].kernel_size, (1, 1))
        self.assertEqual(model.candidate_head[2].out_channels, 13)
        self.assertIsInstance(model.mask_decoder.expand, torch.nn.Conv2d)
        self.assertEqual(model.mask_decoder.expand.out_channels, 16 * 2 * 2)
        self.assertIsInstance(model.mask_decoder.upsample, torch.nn.PixelShuffle)
        self.assertEqual(model.mask_decoder.upsample.upscale_factor, 2)
        self.assertEqual(model.mask_out.kernel_size, (1, 1))
        self.assertEqual(model.mask_out.in_channels, 16)
        self.assertTrue(hasattr(model, "wrapper_blocks"))
        self.assertFalse(any(isinstance(module, torch.nn.PixelUnshuffle) for module in model.modules()))
        self.assertTrue(any(isinstance(module, torch.nn.PixelShuffle) for module in model.modules()))

    def test_wrapper_stack_tracks_configured_inner_block_counts(self) -> None:
        model = LCR(
            dim=32,
            num_blocks=2,
            cross_block_count=3,
            self_block_count=2,
            encoder_block_count=2,
            heads=4,
        )

        self.assertEqual(model.num_blocks, 2)
        self.assertEqual(model.cross_block_count, 3)
        self.assertEqual(model.self_block_count, 2)
        self.assertEqual(model.encoder_block_count, 2)
        self.assertEqual(len(model.wrapper_blocks), 2)
        self.assertTrue(all(isinstance(wrapper, LCRWrapperBlock) for wrapper in model.wrapper_blocks))
        self.assertTrue(all(len(wrapper.cross_blocks) == 3 for wrapper in model.wrapper_blocks))
        self.assertTrue(all(len(wrapper.self_blocks) == 2 for wrapper in model.wrapper_blocks))
        self.assertEqual(len(model.sar_encoder.encoder_blocks), 2)
        self.assertEqual(len(model.hsi_encoder.encoder_blocks), 2)

    def test_lcr_patch_embed_kernel_tracks_patch_size(self) -> None:
        model = LCR(dim=32, num_blocks=1, heads=4, patch_size=3)

        self.assertEqual(model.sar_encoder.patch_embed.kernel_size, (3, 3))
        self.assertEqual(model.sar_encoder.patch_embed.stride, (3, 3))
        self.assertEqual(model.hsi_encoder.patch_embed.kernel_size, (3, 3))
        self.assertEqual(model.candidate_decoder.upsample.upscale_factor, 3)
        self.assertEqual(model.candidate_decoder.expand.out_channels, 16 * 3 * 3)

    def test_lcr_uses_3x3_depthwise_spatial_mixing_only_in_encoder(self) -> None:
        model = LCR(dim=32, num_blocks=1, heads=4)

        convs_3x3 = [
            module
            for module in model.modules()
            if isinstance(module, torch.nn.Conv2d)
            and module.kernel_size == (3, 3)
        ]
        self.assertTrue(convs_3x3)
        self.assertTrue(all(conv.groups == conv.in_channels == conv.out_channels for conv in convs_3x3))
        self.assertEqual(len(convs_3x3), model.encoder_block_count * 2)

    def test_attn_block_uses_same_resolution_attn(self) -> None:
        block = AttnBlock(dim=32, heads=4, ffn_expansion=2)
        self.assertIsInstance(block.attn, Attn)

    def test_attn_block_accepts_z_only_input(self) -> None:
        block = AttnBlock(dim=32, heads=4, ffn_expansion=2)
        z = torch.randn(1, 32, 8, 8)

        out = block(z)

        self.assertEqual(out.shape, z.shape)

    def test_latent_decoder_pixelshuffle_restores_patch_resolution(self) -> None:
        decoder = LatentDecoder(dim=8, patch_size=4, ffn_expansion=2)
        latent = torch.randn(1, 8, 5, 7)

        decoded = decoder(latent)

        self.assertEqual(decoder.patch_size, 4)
        self.assertEqual(decoder.full_dim, 4)
        self.assertEqual(decoder.expand.out_channels, 4 * 4 * 4)
        self.assertIsInstance(decoder.upsample, torch.nn.PixelShuffle)
        self.assertEqual(decoder.upsample.upscale_factor, 4)
        self.assertEqual(decoded.shape, (1, 4, 20, 28))

    def test_mask_out_bias_uses_mask_bias_init(self) -> None:
        model = LCR(dim=32, num_blocks=1, heads=4, mask_bias_init=-1.5)

        self.assertTrue(torch.allclose(model.mask_out.bias, torch.full_like(model.mask_out.bias, -1.5)))


if __name__ == "__main__":
    unittest.main()
