from __future__ import annotations

import copy
import json
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from unittest import mock

import torch
from torch import nn

import main


def make_epoch_record(
    *,
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_metrics: dict[str, float] | None = None,
    train_lrs: Sequence[float] | None = None,
) -> dict[str, object]:
    train_summary: dict[str, object] = {"loss": train_loss, "metrics": {}}
    if train_lrs is not None:
        train_summary["lr"] = list(train_lrs)

    return {
        "epoch": epoch,
        "train": train_summary,
        "val": {"loss": val_loss, "metrics": dict(val_metrics or {})},
        "elapsed_sec": 1.0,
    }


class FakeTrainer:
    step_records: list[dict[str, object]] = []
    test_record: dict[str, object] = {"loss": 0.25, "metrics": {"mae": 0.1}}
    instances: list["FakeTrainer"] = []
    saved_checkpoint_paths: list[Path] = []
    loaded_checkpoint_paths: list[Path] = []

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss,
        metrics,
        scheduler,
        scheduler_timing,
        scheduler_monitor,
        max_train_samples,
        max_val_samples,
        max_test_samples,
        output_dir,
        cache_dir,
        batch_size,
        accum_steps,
        epochs,
        seed,
        num_workers,
        multiprocessing_context,
        train_crop_size,
        train_random_flip,
        train_random_rot90,
    ) -> None:
        del loss, metrics
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_timing = scheduler_timing
        self.scheduler_monitor = scheduler_monitor
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.accum_steps = accum_steps
        self.epochs = epochs
        self.seed = seed
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.max_test_samples = max_test_samples
        self.multiprocessing_context = multiprocessing_context
        self.train_crop_size = train_crop_size
        self.train_random_flip = train_random_flip
        self.train_random_rot90 = train_random_rot90
        self.num_workers = 0 if num_workers == "auto" else int(num_workers)
        self.cache_root = Path(cache_dir) if cache_dir is not None else self.output_dir / "cache"
        self.include_metadata = True
        self.pin_memory = False
        self.persistent_workers = False
        self.prefetch_factor = 2
        self.drop_last = False
        self.current_epoch = 0
        self.global_step = 0
        self.test_calls = 0
        type(self).instances.append(self)

    def step(self) -> dict[str, object]:
        record = copy.deepcopy(type(self).step_records[self.current_epoch])
        epoch = int(record["epoch"])
        self.current_epoch = epoch
        self.global_step = epoch * 10
        return record

    def test(self) -> dict[str, object]:
        self.test_calls += 1
        return copy.deepcopy(type(self).test_record)

    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        checkpoint_path = Path(path) if path is not None else self.output_dir / f"epoch-{self.current_epoch:04d}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(f"checkpoint:{self.current_epoch}\n", encoding="utf-8")
        type(self).saved_checkpoint_paths.append(checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, path: str | Path) -> dict[str, object]:
        checkpoint_path = Path(path)
        payload = checkpoint_path.read_text(encoding="utf-8").strip()
        epoch = int(payload.split(":", maxsplit=1)[1]) if payload.startswith("checkpoint:") else 0
        self.current_epoch = epoch
        self.global_step = epoch * 10
        type(self).loaded_checkpoint_paths.append(checkpoint_path)
        return {
            "path": checkpoint_path,
            "epoch": self.current_epoch,
            "global_step": self.global_step,
        }


class HistoryRecordTest(unittest.TestCase):
    @staticmethod
    def _make_axis() -> mock.Mock:
        axis = mock.Mock()
        axis.spines = {
            "top": mock.Mock(),
            "right": mock.Mock(),
        }
        return axis

    def test_flatten_record_keeps_single_scheduler_learning_rate(self) -> None:
        row = main.flatten_record(
            make_epoch_record(
                epoch=2,
                train_loss=0.8,
                val_loss=0.4,
                train_lrs=[1e-3],
            ),
            global_step=20,
        )

        self.assertEqual(row["train_lr"], 1e-3)
        self.assertNotIn("train_lr_group_0", row)

    def test_load_history_from_metrics_jsonl_restores_multi_group_learning_rates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "metrics.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "kind": "train_epoch",
                                "epoch": 1,
                                "loss": 1.2,
                                "metrics": {},
                                "lr": [1e-3, 5e-4],
                            }
                        ),
                        json.dumps(
                            {
                                "kind": "validation",
                                "epoch": 1,
                                "loss": 0.5,
                                "metrics": {"mae": 0.6},
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            history = main.load_history_from_metrics_jsonl(path)

        self.assertEqual(
            history,
            [
                {
                    "epoch": 1,
                    "global_step": 1,
                    "train_loss": 1.2,
                    "train_lr_group_0": 1e-3,
                    "train_lr_group_1": 5e-4,
                    "val_loss": 0.5,
                    "val_mae": 0.6,
                }
            ],
        )

    def test_save_history_plot_writes_loss_lr_then_metric_plots(self) -> None:
        history = [
            {
                "epoch": 1,
                "global_step": 10,
                "elapsed_sec": 99.0,
                "train_loss": 1.2,
                "train_lr": 1e-3,
                "val_loss": 0.5,
                "val_mae": 0.6,
            },
            {
                "epoch": 2,
                "global_step": 20,
                "elapsed_sec": 101.0,
                "train_loss": 1.0,
                "train_lr": 5e-4,
                "val_loss": 0.4,
                "val_mae": 0.5,
            },
        ]

        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            mock.patch.object(main, "_save_metric_plot") as save_metric_plot,
        ):
            path = Path(tmp_dir) / "history.png"
            main.save_history_plot(history, path)

        self.assertEqual(
            [call.kwargs["metric"] for call in save_metric_plot.call_args_list],
            ["loss", "lr", "mae"],
        )
        self.assertTrue(all(call.kwargs["path"] == path for call in save_metric_plot.call_args_list))

    def test_save_metric_plot_uses_lr_group_label_and_filename(self) -> None:
        history = [
            {
                "epoch": 1,
                "global_step": 10,
                "train_lr_group_0": 1e-3,
                "train_lr_group_1": 5e-4,
            },
            {
                "epoch": 2,
                "global_step": 20,
                "train_lr_group_0": 8e-4,
                "train_lr_group_1": 4e-4,
            },
        ]
        fig = mock.Mock()
        axis = self._make_axis()

        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            mock.patch("matplotlib.pyplot.subplots", return_value=(fig, axis)),
            mock.patch("matplotlib.pyplot.close") as close_plot,
        ):
            path = Path(tmp_dir) / "history.png"
            main._save_metric_plot(history, path=path, metric="lr_group_1")

        axis.set_title.assert_called_once_with("LR (Group 1)")
        axis.plot.assert_called_once()
        self.assertEqual(axis.plot.call_args.args[0], (1, 2))
        self.assertEqual(axis.plot.call_args.args[1], (5e-4, 4e-4))
        self.assertEqual(Path(fig.savefig.call_args.args[0]), Path(tmp_dir) / "history_lr_group_1.png")
        close_plot.assert_called_once_with(fig)


class MainTrainingFlowTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeTrainer.instances = []
        FakeTrainer.saved_checkpoint_paths = []
        FakeTrainer.loaded_checkpoint_paths = []

    def test_default_best_selector_uses_validation_loss(self) -> None:
        selector = main.build_best_epoch_selector()

        self.assertEqual(selector.name, "val_loss")
        self.assertEqual(selector.mode, "min")
        self.assertEqual(
            main.score_epoch(
                {
                    "epoch": 1,
                    "val": {"loss": 0.125, "metrics": {"mae": 0.3}},
                },
                selector=selector,
            ),
            0.125,
        )

    def test_load_best_state_ignores_selector_mismatch(self) -> None:
        selector = main.BestEpochSelector(name="val_loss", mode="min", score_fn=lambda record: 0.0)
        mismatched = main.BestEpochSelector(name="val_mae", mode="min", score_fn=lambda record: 0.0)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            main.best_metadata_path(output_dir).write_text(
                json.dumps(
                    {
                        "epoch": 3,
                        "score": 0.2,
                        "selector_name": selector.name,
                        "selector_mode": selector.mode,
                        "source_checkpoint_path": "epoch-0003.pt",
                    }
                ),
                encoding="utf-8",
            )

            best_state = main.load_best_state(output_dir, selector=selector)
            self.assertIsNotNone(best_state)
            self.assertEqual(best_state.epoch, 3)
            self.assertEqual(best_state.score, 0.2)

            self.assertIsNone(main.load_best_state(output_dir, selector=mismatched))

    def test_best_mode_updates_best_checkpoint_and_saves_examples_on_improvement(self) -> None:
        records = [
            make_epoch_record(epoch=1, train_loss=1.2, val_loss=0.5, val_metrics={"mae": 0.6}),
            make_epoch_record(epoch=2, train_loss=1.0, val_loss=0.4, val_metrics={"mae": 0.5}),
            make_epoch_record(epoch=3, train_loss=0.8, val_loss=0.45, val_metrics={"mae": 0.4}),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            result = self._run_main(
                output_dir=output_dir,
                step_records=records,
                example_mode="best",
                run_test=False,
                num_examples=2,
            )

            best_payload = json.loads((output_dir / "best.json").read_text(encoding="utf-8"))
            self.assertEqual(best_payload["epoch"], 2)
            self.assertEqual(best_payload["checkpoint_path"], str(output_dir / "best.pt"))
            self.assertEqual(best_payload["selector_name"], "val_loss")
            self.assertEqual(best_payload["selector_mode"], "min")
            self.assertAlmostEqual(best_payload["score"], 0.4)
            self.assertEqual((output_dir / "best.pt").read_text(encoding="utf-8"), "checkpoint:2\n")
            self.assertFalse((output_dir / "epoch-0001.pt").exists())
            self.assertFalse((output_dir / "epoch-0002.pt").exists())
            self.assertFalse((output_dir / "epoch-0003.pt").exists())

            self.assertEqual(result["trainer"].test_calls, 0)
            self.assertEqual(result["save_examples"].call_count, 2)
            first_output_dir = result["save_examples"].call_args_list[0].kwargs["output_dir"]
            second_output_dir = result["save_examples"].call_args_list[1].kwargs["output_dir"]
            self.assertEqual(first_output_dir, output_dir / "examples" / "epoch_001" / "test")
            self.assertEqual(second_output_dir, output_dir / "examples" / "epoch_002" / "test")
            self.assertEqual(result["save_examples"].call_args_list[0].kwargs["split"], "test")
            self.assertEqual(result["save_examples"].call_args_list[1].kwargs["split"], "test")
            self.assertEqual(result["save_examples"].call_args_list[0].kwargs["epoch"], 1)
            self.assertEqual(result["save_examples"].call_args_list[1].kwargs["epoch"], 2)

    def test_after_test_mode_runs_test_once_and_saves_test_examples(self) -> None:
        records = [
            make_epoch_record(epoch=1, train_loss=1.0, val_loss=0.6),
            make_epoch_record(epoch=2, train_loss=0.9, val_loss=0.5),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            result = self._run_main(
                output_dir=output_dir,
                step_records=records,
                example_mode="after_test",
                run_test=False,
                num_examples=3,
            )

            self.assertEqual(result["trainer"].test_calls, 1)
            self.assertEqual(result["save_examples"].call_count, 1)
            self.assertEqual(result["save_examples"].call_args.kwargs["split"], "test")
            self.assertEqual(result["save_examples"].call_args.kwargs["epoch"], 2)
            self.assertEqual(
                result["save_examples"].call_args.kwargs["output_dir"],
                output_dir / "examples" / "epoch_002" / "test",
            )

    def test_build_trainer_passes_scheduler_options_to_trainer(self) -> None:
        model = nn.Linear(1, 1)
        scheduler = object()

        with (
            mock.patch.object(main, "Trainer", FakeTrainer),
            mock.patch.object(main, "build_model", return_value=model),
            mock.patch.object(main, "build_optimizer", side_effect=lambda current_model: torch.optim.SGD(current_model.parameters(), lr=0.1)),
            mock.patch.object(main, "build_loss", return_value=lambda prediction, batch: torch.tensor(0.0)),
            mock.patch.object(main, "build_metrics", return_value={}),
            mock.patch.object(main, "build_scheduler", return_value=scheduler),
            mock.patch.object(main, "build_scheduler_monitor", return_value="val.metrics.mae"),
            mock.patch.object(main, "seed_everything"),
            mock.patch.object(main, "print_hf_auth_status"),
            mock.patch.object(main.torch.cuda, "is_available", return_value=False),
        ):
            trainer = main.build_trainer(
                output_dir=Path("artifacts"),
                accum_steps=4,
                scheduler_timing="after_optimizer_step",
            )

        self.assertIs(trainer.scheduler, scheduler)
        self.assertEqual(trainer.accum_steps, 4)
        self.assertEqual(trainer.scheduler_timing, "after_optimizer_step")
        self.assertEqual(trainer.scheduler_monitor, "val.metrics.mae")

    def test_main_preserves_sample_limits(self) -> None:
        records = [
            make_epoch_record(epoch=1, train_loss=1.0, val_loss=0.6),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            result = self._run_main(
                output_dir=output_dir,
                step_records=records,
                example_mode="best",
                run_test=False,
                num_examples=2,
                train_max_samples=128,
                val_max_samples=64,
                test_max_samples=32,
            )

        trainer = result["trainer"]
        self.assertEqual(trainer.max_train_samples, 128)
        self.assertEqual(trainer.max_val_samples, 64)
        self.assertEqual(trainer.max_test_samples, 32)
        self.assertEqual(result["save_examples"].call_args.kwargs["max_samples"], 32)

    def test_custom_best_selector_can_change_best_epoch(self) -> None:
        records = [
            make_epoch_record(epoch=1, train_loss=1.2, val_loss=0.3, val_metrics={"score": 0.2}),
            make_epoch_record(epoch=2, train_loss=1.1, val_loss=0.4, val_metrics={"score": 0.8}),
            make_epoch_record(epoch=3, train_loss=1.0, val_loss=0.2, val_metrics={"score": 0.5}),
        ]
        selector = main.BestEpochSelector(
            name="val_score",
            mode="max",
            score_fn=lambda record: float(record["val"]["metrics"]["score"]),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            self._run_main(
                output_dir=output_dir,
                step_records=records,
                example_mode="best",
                run_test=False,
                num_examples=0,
                selector=selector,
            )

            best_payload = json.loads((output_dir / "best.json").read_text(encoding="utf-8"))
            self.assertEqual(best_payload["epoch"], 2)
            self.assertEqual(best_payload["selector_name"], "val_score")
            self.assertEqual(best_payload["selector_mode"], "max")
            self.assertAlmostEqual(best_payload["score"], 0.8)
            self.assertEqual((output_dir / "best.pt").read_text(encoding="utf-8"), "checkpoint:2\n")

    def test_best_mode_runs_test_by_default(self) -> None:
        records = [
            make_epoch_record(epoch=1, train_loss=1.0, val_loss=0.6),
            make_epoch_record(epoch=2, train_loss=0.9, val_loss=0.5),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            result = self._run_main(
                output_dir=output_dir,
                step_records=records,
                example_mode="best",
                run_test=None,
                num_examples=0,
            )

            self.assertEqual(result["trainer"].test_calls, 1)

    def test_save_every_n_epochs_saves_periodic_checkpoints(self) -> None:
        records = [
            make_epoch_record(epoch=1, train_loss=1.0, val_loss=0.5),
            make_epoch_record(epoch=2, train_loss=0.9, val_loss=0.6),
            make_epoch_record(epoch=3, train_loss=0.8, val_loss=0.7),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            self._run_main(
                output_dir=output_dir,
                step_records=records,
                example_mode="best",
                run_test=False,
                num_examples=0,
                save_every_n_epochs=2,
            )

            self.assertTrue((output_dir / "best.pt").exists())
            self.assertTrue((output_dir / "epoch-0002.pt").exists())
            self.assertFalse((output_dir / "epoch-0001.pt").exists())
            self.assertFalse((output_dir / "epoch-0003.pt").exists())

    def test_resume_uses_trainer_load_checkpoint(self) -> None:
        records = [
            make_epoch_record(epoch=1, train_loss=1.0, val_loss=0.6),
            make_epoch_record(epoch=2, train_loss=0.9, val_loss=0.5),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            resume_path = output_dir / "resume.pt"
            resume_path.write_text("checkpoint:1\n", encoding="utf-8")

            result = self._run_main(
                output_dir=output_dir,
                step_records=records,
                example_mode="best",
                run_test=False,
                num_examples=0,
                resume=resume_path,
            )

            self.assertEqual(FakeTrainer.loaded_checkpoint_paths, [resume_path])
            self.assertEqual(result["trainer"].current_epoch, 2)

    def _run_main(
        self,
        *,
        output_dir: Path,
        step_records: list[dict[str, object]],
        example_mode: str,
        num_examples: int,
        run_test: bool | None = False,
        selector: main.BestEpochSelector | None = None,
        save_every_n_epochs: int = 0,
        resume: Path | None = None,
        train_max_samples: int | None = 16384,
        val_max_samples: int | None = 2048,
        test_max_samples: int | None = 2048,
    ) -> dict[str, object]:
        FakeTrainer.step_records = copy.deepcopy(step_records)
        FakeTrainer.instances = []
        FakeTrainer.test_record = {"loss": 0.25, "metrics": {"mae": 0.1}}
        FakeTrainer.saved_checkpoint_paths = []
        FakeTrainer.loaded_checkpoint_paths = []

        build_loader = mock.Mock(side_effect=lambda *args, **kwargs: {"loader": kwargs})
        save_examples = mock.Mock(return_value=[])
        history_plot = mock.Mock()
        build_scheduler = mock.Mock(return_value=None)
        build_scheduler_monitor = mock.Mock(return_value=None)

        model = nn.Linear(1, 1)
        if selector is None:
            selector = main.build_best_epoch_selector()

        with (
            mock.patch.object(main, "Trainer", FakeTrainer),
            mock.patch.object(main, "build_model", return_value=model),
            mock.patch.object(main, "build_optimizer", side_effect=lambda current_model: torch.optim.SGD(current_model.parameters(), lr=0.1)),
            mock.patch.object(main, "build_loss", return_value=lambda prediction, batch: torch.tensor(0.0)),
            mock.patch.object(main, "build_metrics", return_value={}),
            mock.patch.object(main, "build_scheduler", build_scheduler),
            mock.patch.object(main, "build_scheduler_monitor", build_scheduler_monitor),
            mock.patch.object(main, "build_best_epoch_selector", return_value=selector),
            mock.patch.object(main, "build_loader", build_loader),
            mock.patch.object(main, "save_restoration_examples", save_examples),
            mock.patch.object(main, "save_history_plot", history_plot),
            mock.patch.object(main, "seed_everything"),
            mock.patch.object(main, "print_hf_auth_status"),
            mock.patch.object(main.torch.cuda, "is_available", return_value=False),
        ):
            kwargs = {
                "output_dir": output_dir,
                "max_epochs": len(step_records),
                "example_mode": example_mode,
                "num_examples": num_examples,
                "save_every_n_epochs": save_every_n_epochs,
                "train_max_samples": train_max_samples,
                "val_max_samples": val_max_samples,
                "test_max_samples": test_max_samples,
            }
            if run_test is not None:
                kwargs["run_test"] = run_test
            if resume is not None:
                kwargs["resume"] = resume

            main.main(
                **kwargs,
            )

        return {
            "trainer": FakeTrainer.instances[-1],
            "build_loader": build_loader,
            "build_scheduler": build_scheduler,
            "build_scheduler_monitor": build_scheduler_monitor,
            "save_examples": save_examples,
            "history_plot": history_plot,
        }


class BuildLoaderTest(unittest.TestCase):
    def _make_trainer(self) -> mock.Mock:
        trainer = mock.Mock()
        trainer.seed = 7
        trainer.cache_root = Path("/tmp/cache-root")
        trainer.batch_size = 4
        trainer.num_workers = 2
        trainer.include_metadata = True
        trainer.pin_memory = False
        trainer.multiprocessing_context = None
        trainer.persistent_workers = False
        trainer.prefetch_factor = 2
        trainer.drop_last = False
        trainer.train_crop_size = 128
        trainer.train_random_flip = True
        trainer.train_random_rot90 = True
        return trainer

    def test_build_loader_warms_cache_before_prepare_split(self) -> None:
        trainer = self._make_trainer()
        prepared = object()

        with (
            mock.patch.object(main, "ensure_split_cache") as ensure_split_cache,
            mock.patch.object(main, "prepare_split", return_value=prepared) as prepare_split,
            mock.patch.object(main, "build_dataloader", return_value="loader") as build_dataloader,
        ):
            result = main.build_loader(
                trainer,
                split="validation",
                max_samples=None,
                training=False,
                epoch_index=3,
            )

        self.assertEqual(result, "loader")
        ensure_split_cache.assert_called_once()
        self.assertNotIn("streaming", prepare_split.call_args.kwargs)
        self.assertEqual(prepare_split.call_args.kwargs["max_samples"], None)
        self.assertEqual(build_dataloader.call_args.kwargs["crop_size"], None)
        self.assertEqual(build_dataloader.call_args.kwargs["crop_mode"], "none")

    def test_build_loader_passes_training_options(self) -> None:
        trainer = self._make_trainer()
        prepared = object()

        with (
            mock.patch.object(main, "ensure_split_cache") as ensure_split_cache,
            mock.patch.object(main, "prepare_split", return_value=prepared) as prepare_split,
            mock.patch.object(main, "build_dataloader", return_value="loader"),
        ):
            main.build_loader(
                trainer,
                split="train",
                max_samples=256,
                training=True,
                epoch_index=1,
            )

        ensure_split_cache.assert_called_once()
        self.assertNotIn("streaming", prepare_split.call_args.kwargs)
        self.assertEqual(prepare_split.call_args.kwargs["max_samples"], 256)


if __name__ == "__main__":
    unittest.main()
