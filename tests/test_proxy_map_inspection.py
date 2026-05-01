from __future__ import annotations

import unittest

import numpy as np
import torch

from scripts.inspect_proxy_map import blend_overlay, normalize_proxy_map


class ProxyMapInspectionTest(unittest.TestCase):
    def test_normalize_proxy_map_uses_percentile_range(self) -> None:
        proxy = torch.tensor(
            [
                [0.0, 1.0],
                [2.0, 100.0],
            ]
        )
        normalized, stats = normalize_proxy_map(proxy, low_percentile=0.0, high_percentile=75.0)
        self.assertEqual(normalized.shape, (2, 2))
        self.assertAlmostEqual(float(normalized[0, 0]), 0.0)
        self.assertAlmostEqual(float(normalized[1, 1]), 1.0)
        self.assertAlmostEqual(stats["min"], 0.0)
        self.assertAlmostEqual(stats["max"], 100.0)

    def test_blend_overlay_preserves_shape_and_range(self) -> None:
        base = np.zeros((4, 4, 3), dtype=np.float32)
        heatmap = np.ones((4, 4, 3), dtype=np.float32)
        overlay = blend_overlay(base, heatmap, alpha=0.25)
        self.assertEqual(overlay.shape, base.shape)
        self.assertTrue(np.allclose(overlay, 0.25))
        self.assertGreaterEqual(float(overlay.min()), 0.0)
        self.assertLessEqual(float(overlay.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
