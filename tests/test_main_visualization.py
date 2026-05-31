from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from main import _save_example_figure, normalize_rgb_triplet
from modules.util.fdt_visualization import (
    _weighted_pca_map,
    build_fdt_example_panels,
    load_fdt_checkpoint_model,
    save_fdt_tsne_scatter,
)


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

    def test_fdt_example_panels_use_pca_mask_candidate_rows(self) -> None:
        def normalize_triplet(*_):
            return tuple(np.zeros((4, 4, 3), dtype=np.float32) for _ in range(3))

        def normalize_map(tensor):
            return np.zeros(tuple(tensor.shape), dtype=np.float32)

        feature = torch.randn(8, 4, 4)
        mask = torch.full((13, 4, 4), 0.25)
        model_output = {
            "prediction": torch.zeros(13, 4, 4),
            "candidate": torch.zeros(13, 4, 4),
            "mask": mask,
            "sar_feat": feature,
            "clear_feat": feature.roll(shifts=1, dims=-1),
            "cloud_feat": torch.randn(8, 4, 4),
        }

        panels = build_fdt_example_panels(
            cloudy=torch.zeros(13, 4, 4),
            prediction=torch.zeros(13, 4, 4),
            target=torch.zeros(13, 4, 4),
            sar=torch.zeros(2, 4, 4),
            model_output=model_output,
            normalize_rgb_triplet=normalize_triplet,
            normalize_map=normalize_map,
        )

        self.assertEqual(
            [panel[0] for panel in panels],
            [
                "Cloudy RGB",
                "Prediction RGB",
                "Target RGB",
                "SAR Mean",
                "Mask PCA",
                "Prediction PCA",
                "Target PCA",
                "Candidate RGB",
                "SAR Feat PCA",
                "CLD Feat PCA",
                "CLD Clean PCA",
                "CLD Cloudy PCA",
            ],
        )
        for index in (4, 5, 6, 8, 9, 10, 11):
            self.assertEqual(panels[index][2], "magma")

    def test_weighted_pca_map_uses_variance_weighted_components(self) -> None:
        feature = torch.tensor(
            [
                [[1.0, -1.0], [0.0, 0.0]],
                [[0.0, 0.0], [2.0, -2.0]],
            ]
        )

        def identity_map(tensor):
            return tensor.numpy()

        actual = _weighted_pca_map(feature, identity_map)
        expected = np.array(
            [
                [np.sqrt(0.2), -np.sqrt(0.2)],
                [2.0 * np.sqrt(0.8), -2.0 * np.sqrt(0.8)],
            ],
            dtype=np.float32,
        )

        self.assertTrue(np.allclose(actual, expected, atol=1e-6))

    def test_fdt_tsne_scatter_uses_dataloader_and_predict_fn(self) -> None:
        captured = {}

        def fake_save(grouped_features, **kwargs):
            captured["features"] = grouped_features
            captured["kwargs"] = kwargs
            return kwargs["path"]

        def predict_fn(batch):
            value = float(batch["value"])
            feature = torch.full((1, 2, 2, 2), value)
            candidate = torch.full((1, 13, 2, 2), value)
            mask = torch.full((1, 13, 2, 2), 0.25)
            return {
                "prediction": feature,
                "candidate": candidate,
                "mask": mask,
                "sar_feat": feature + 1.0,
                "clear_feat": feature + 2.0,
                "cloud_feat": feature + 3.0,
            }

        dataloader = [{"value": 1.0}, {"value": 2.0}]

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tsne.png"
            with patch(
                "modules.util.fdt_visualization._save_tsne_scatter_figure",
                new=fake_save,
            ):
                actual = save_fdt_tsne_scatter(
                    dataloader=dataloader,
                    predict_fn=predict_fn,
                    path=path,
                    title="test",
                    is_primary=lambda: True,
                    sample_count=2,
                    max_points_per_group=8,
                    show_progress=False,
                )

        self.assertEqual(actual, path)
        self.assertEqual(captured["kwargs"]["title"], "test")
        self.assertEqual(
            set(captured["features"]),
            {"SAR Feat", "CLD Feat", "CLD Clean", "CLD Cloudy"},
        )
        for features in captured["features"].values():
            self.assertEqual(features.shape, (8, 2))

    def test_load_fdt_checkpoint_model_loads_only_model_state(self) -> None:
        source = nn.Linear(2, 1)
        target = nn.Linear(2, 1)
        with torch.no_grad():
            source.weight.fill_(3.0)
            source.bias.fill_(0.5)

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "best.pt"
            torch.save({"epoch": 7, "model": source.state_dict(), "optimizer": {}}, path)
            epoch = load_fdt_checkpoint_model(target, path, torch.device("cpu"))

        self.assertEqual(epoch, 7)
        self.assertTrue(torch.allclose(target.weight, source.weight))
        self.assertTrue(torch.allclose(target.bias, source.bias))


if __name__ == "__main__":
    unittest.main()
