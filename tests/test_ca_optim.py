from __future__ import annotations

import unittest

import torch

from modules.model.baseline.ca import ConAttn as OriginalConAttn
from modules.model.baseline.ca_optim import ConAttn as OptimConAttn


class OptimConAttnTest(unittest.TestCase):
    def test_default_and_auto_forward_matches_original_for_supported_path(self) -> None:
        torch.manual_seed(5)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        original = OriginalConAttn(input_channels=8, output_channels=8, ksize=1, stride=1).to(device)
        x = torch.randn(2, 8, 3, 4, device=device)
        expected = original(x)

        for kwargs in ({}, {"chunk_size": "auto"}):
            with self.subTest(chunk_size=kwargs.get("chunk_size", "default")):
                optimized = OptimConAttn(input_channels=8, output_channels=8, ksize=1, stride=1, **kwargs).to(device)
                optimized.load_state_dict(original.state_dict())

                actual = optimized(x)

                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_forward_matches_original_for_supported_path(self) -> None:
        torch.manual_seed(7)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        original = OriginalConAttn(input_channels=8, output_channels=8, ksize=1, stride=1).to(device)
        optimized = OptimConAttn(input_channels=8, output_channels=8, ksize=1, stride=1, chunk_size=3).to(device)
        optimized.load_state_dict(original.state_dict())

        x = torch.randn(2, 8, 3, 4, device=device)

        expected = original(x)
        actual = optimized(x)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_backward_matches_original_for_supported_path(self) -> None:
        torch.manual_seed(11)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        original = OriginalConAttn(input_channels=8, output_channels=8, ksize=1, stride=1).to(device)
        optimized = OptimConAttn(input_channels=8, output_channels=8, ksize=1, stride=1, chunk_size=4).to(device)
        optimized.load_state_dict(original.state_dict())

        original_x = torch.randn(1, 8, 3, 3, device=device, requires_grad=True)
        optimized_x = original_x.detach().clone().requires_grad_(True)

        original(original_x).sum().backward()
        optimized(optimized_x).sum().backward()

        torch.testing.assert_close(optimized_x.grad, original_x.grad, rtol=1e-5, atol=1e-6)
        for (original_name, original_param), (optimized_name, optimized_param) in zip(
            original.named_parameters(),
            optimized.named_parameters(),
        ):
            self.assertEqual(optimized_name, original_name)
            torch.testing.assert_close(optimized_param.grad, original_param.grad, rtol=1e-5, atol=1e-6)

    def test_auto_backward_matches_original_for_supported_path(self) -> None:
        torch.manual_seed(13)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        original = OriginalConAttn(input_channels=8, output_channels=8, ksize=1, stride=1).to(device)
        optimized = OptimConAttn(input_channels=8, output_channels=8, ksize=1, stride=1, chunk_size="auto").to(device)
        optimized.load_state_dict(original.state_dict())

        original_x = torch.randn(1, 8, 3, 3, device=device, requires_grad=True)
        optimized_x = original_x.detach().clone().requires_grad_(True)

        original(original_x).sum().backward()
        optimized(optimized_x).sum().backward()

        torch.testing.assert_close(optimized_x.grad, original_x.grad, rtol=1e-5, atol=1e-6)
        for (original_name, original_param), (optimized_name, optimized_param) in zip(
            original.named_parameters(),
            optimized.named_parameters(),
        ):
            self.assertEqual(optimized_name, original_name)
            torch.testing.assert_close(optimized_param.grad, original_param.grad, rtol=1e-5, atol=1e-6)

    def test_rejects_unsupported_kernel_or_stride(self) -> None:
        with self.assertRaisesRegex(ValueError, "ksize=1"):
            OptimConAttn(input_channels=8, output_channels=8, ksize=3, stride=1)

        with self.assertRaisesRegex(ValueError, "stride=1"):
            OptimConAttn(input_channels=8, output_channels=8, ksize=1, stride=2)

    def test_rejects_non_positive_chunk_size(self) -> None:
        for chunk_size in (0, -1):
            with self.subTest(chunk_size=chunk_size):
                with self.assertRaisesRegex(ValueError, "chunk_size"):
                    OptimConAttn(input_channels=8, output_channels=8, chunk_size=chunk_size)

    def test_rejects_invalid_chunk_size_string(self) -> None:
        with self.assertRaisesRegex(ValueError, "chunk_size"):
            OptimConAttn(input_channels=8, output_channels=8, chunk_size="dynamic")


if __name__ == "__main__":
    unittest.main()
