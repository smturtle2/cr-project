from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from main import _save_example_figure, normalize_rgb_triplet


class MainVisualizationTest(unittest.TestCase):
    def test_cloudy_and_target_visualization_do_not_depend_on_prediction(self) -> None:
        cloudy = torch.zeros(13, 8, 8)
        cloudy[3] = 0.1
        cloudy[2] = 0.2
        cloudy[1] = 0.3

        target = torch.zeros(13, 8, 8)
        target[3] = 0.2
        target[2] = 0.4
        target[1] = 0.6

        prediction_a = target.clone()
        prediction_b = torch.full_like(target, 100.0)
        prediction_c = torch.full_like(target, -100.0)

        cloudy_rgb_a, _, target_rgb_a = normalize_rgb_triplet(cloudy, prediction_a, target)
        cloudy_rgb_b, prediction_rgb_b, target_rgb_b = normalize_rgb_triplet(cloudy, prediction_b, target)
        cloudy_rgb_c, prediction_rgb_c, target_rgb_c = normalize_rgb_triplet(cloudy, prediction_c, target)

        self.assertTrue(np.allclose(cloudy_rgb_a, cloudy_rgb_b))
        self.assertTrue(np.allclose(cloudy_rgb_a, cloudy_rgb_c))
        self.assertTrue(np.allclose(target_rgb_a, target_rgb_b))
        self.assertTrue(np.allclose(target_rgb_a, target_rgb_c))
        self.assertTrue(np.isfinite(prediction_rgb_b).all())
        self.assertTrue(np.isfinite(prediction_rgb_c).all())
        self.assertGreaterEqual(float(prediction_rgb_b.min()), 0.0)
        self.assertGreaterEqual(float(prediction_rgb_c.min()), 0.0)
        self.assertLessEqual(float(prediction_rgb_b.max()), 1.0)
        self.assertLessEqual(float(prediction_rgb_c.max()), 1.0)

    def test_example_figure_accepts_fixed_color_scale_panel(self) -> None:
        captured_kwargs: list[dict] = []

        def record_imshow(self, image, **kwargs):
            del self, image
            captured_kwargs.append(kwargs)
            return None

        panels = (
            ("RGB", np.zeros((2, 2, 3)), None),
            ("Corr", np.zeros((2, 2)), "coolwarm", -1.0, 1.0),
        )

        with TemporaryDirectory() as tmpdir:
            with patch("matplotlib.axes.Axes.imshow", new=record_imshow):
                path = _save_example_figure(
                    output_dir=Path(tmpdir),
                    split_label="test",
                    example_index=1,
                    title="example",
                    panels=panels,
                )
                self.assertTrue(path.exists())

        self.assertEqual(captured_kwargs[0], {})
        self.assertEqual(captured_kwargs[1]["cmap"], "coolwarm")
        self.assertEqual(captured_kwargs[1]["vmin"], -1.0)
        self.assertEqual(captured_kwargs[1]["vmax"], 1.0)


if __name__ == "__main__":
    unittest.main()
