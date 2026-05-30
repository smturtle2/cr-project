from __future__ import annotations

import copy
import json
import os
import subprocess
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
        streaming,
        dataset_dir,
        batch_size,
        accum_steps,
        epochs,
        seed,
        num_workers,
        multiprocessing_context,
        train_crop_size,
        train_random_flip,
        train_random_rot90,
        grad_clip_norm,
        mixed_precision,
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
        self.streaming = bool(streaming)
        self.dataset_root = Path(dataset_dir) if dataset_dir is not None else self.output_dir / "dataset"
        self.multiprocessing_context = multiprocessing_context
        self.train_crop_size = train_crop_size
        self.train_random_flip = train_random_flip
        self.train_random_rot90 = train_random_rot90
        self.grad_clip_norm = grad_clip_norm
        self.mixed_precision = mixed_precision
        self.num_workers = 0 if num_workers == "auto" else int(num_workers)
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

    def test_load_resume_history_filters_records_after_checkpoint_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            path = output_dir / "metrics.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"kind": "train_epoch", "epoch": 1, "loss": 1.2, "metrics": {}}),
                        json.dumps({"kind": "validation", "epoch": 1, "loss": 0.5, "metrics": {"mae": 0.6}}),
                        json.dumps({"kind": "train_epoch", "epoch": 2, "loss": 1.0, "metrics": {}}),
                        json.dumps({"kind": "validation", "epoch": 2, "loss": 0.4, "metrics": {"mae": 0.5}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            history = main.load_resume_history(output_dir, current_epoch=1)

        self.assertEqual(
            history,
            [
                {
                    "epoch": 1,
                    "global_step": 1,
                    "train_loss": 1.2,
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
                grad_clip_norm=None,
                streaming=False,
                dataset_dir=Path("/tmp/dataset"),
            )

        self.assertIs(trainer.scheduler, scheduler)
        self.assertEqual(trainer.accum_steps, 4)
        self.assertEqual(trainer.scheduler_timing, "after_optimizer_step")
        self.assertEqual(trainer.scheduler_monitor, "val.metrics.mae")
        self.assertIsNone(trainer.grad_clip_norm)
        self.assertEqual(trainer.mixed_precision, "bf16")
        self.assertFalse(trainer.streaming)
        self.assertEqual(trainer.dataset_root, Path("/tmp/dataset"))

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
            make_epoch_record(epoch=1, train_loss=1.0, val_loss=0.5),
            make_epoch_record(epoch=2, train_loss=0.9, val_loss=0.6),
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
            self.assertEqual(FakeTrainer.loaded_checkpoint_paths, [output_dir / "best.pt"])
            self.assertEqual(result["trainer"].current_epoch, 1)

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

    def test_distributed_epoch_examples_are_saved_on_primary_with_barrier(self) -> None:
        example_config = main.ExampleSaveConfig(
            splits=("test",),
            max_samples_by_split={"test": 16},
            num_examples=2,
        )
        saved_epochs: set[int] = set()

        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            mock.patch.object(main, "is_primary", return_value=True),
            mock.patch.object(main.dist, "is_available", return_value=True),
            mock.patch.object(main.dist, "is_initialized", return_value=True),
            mock.patch.object(main.dist, "barrier") as barrier,
            mock.patch.object(main, "save_examples_for_splits", return_value={"test": []}) as save_examples,
        ):
            output_dir = Path(tmp_dir)
            result = main.maybe_save_epoch_examples(
                mock.Mock(),
                output_dir=output_dir,
                epoch=1,
                example_config=example_config,
                saved_epochs=saved_epochs,
            )
            self.assertFalse((output_dir / ".example_sync").exists())

        self.assertEqual(result, {"test": []})
        self.assertEqual(saved_epochs, {1})
        barrier.assert_called_once_with()
        save_examples.assert_called_once()

    def test_distributed_epoch_examples_barrier_on_non_primary(self) -> None:
        example_config = main.ExampleSaveConfig(
            splits=("test",),
            max_samples_by_split={"test": 16},
            num_examples=2,
        )
        saved_epochs: set[int] = set()

        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            mock.patch.object(main, "is_primary", return_value=False),
            mock.patch.object(main.dist, "is_available", return_value=True),
            mock.patch.object(main.dist, "is_initialized", return_value=True),
            mock.patch.object(main.dist, "barrier") as barrier,
            mock.patch.object(main, "save_examples_for_splits") as save_examples,
        ):
            output_dir = Path(tmp_dir)
            result = main.maybe_save_epoch_examples(
                mock.Mock(),
                output_dir=output_dir,
                epoch=1,
                example_config=example_config,
                saved_epochs=saved_epochs,
            )

        self.assertEqual(result, {})
        self.assertEqual(saved_epochs, {1})
        save_examples.assert_not_called()
        barrier.assert_called_once_with()

    def test_deferred_distributed_examples_load_best_checkpoint(self) -> None:
        selector = main.build_best_epoch_selector()

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            trainer = mock.Mock()
            best_path = output_dir / "best.pt"
            best_path.write_text("checkpoint:2\n", encoding="utf-8")
            main.save_best_state(
                output_dir,
                epoch=2,
                score=0.4,
                selector=selector,
                checkpoint_path=best_path,
            )
            example_config = main.ExampleSaveConfig(
                splits=("validation", "test"),
                max_samples_by_split={"validation": 32, "test": 64},
                num_examples=3,
            )

            with mock.patch.object(main, "save_examples_for_splits", return_value={"test": []}) as save_examples:
                result = main.save_deferred_examples_after_distributed_cleanup(
                    trainer,
                    output_dir=output_dir,
                    best_selector=selector,
                    example_mode="best",
                    example_config=example_config,
                )

        self.assertEqual(result, {"test": []})
        trainer.load_checkpoint.assert_called_once_with(best_path)
        self.assertEqual(save_examples.call_args.kwargs["splits"], ("validation", "test"))
        self.assertEqual(save_examples.call_args.kwargs["epoch"], 2)
        self.assertEqual(
            save_examples.call_args.kwargs["output_dir"],
            output_dir / "examples" / "epoch_002",
        )

    def test_main_saves_distributed_best_examples_before_cleanup(self) -> None:
        records = [
            make_epoch_record(epoch=1, train_loss=1.0, val_loss=0.4),
        ]
        events: list[str] = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            model = nn.Linear(1, 1)
            with (
                mock.patch.object(main, "Trainer", FakeTrainer),
                mock.patch.object(main, "build_model", return_value=model),
                mock.patch.object(main, "build_optimizer", side_effect=lambda current_model: torch.optim.SGD(current_model.parameters(), lr=0.1)),
                mock.patch.object(main, "build_loss", return_value=lambda prediction, batch: torch.tensor(0.0)),
                mock.patch.object(main, "build_metrics", return_value={}),
                mock.patch.object(main, "build_scheduler", return_value=None),
                mock.patch.object(main, "build_scheduler_monitor", return_value=None),
                mock.patch.object(main, "is_distributed_run", return_value=True),
                mock.patch.object(main, "is_primary", return_value=True),
                mock.patch.object(main.dist, "barrier"),
                mock.patch.object(
                    main,
                    "save_examples_for_splits",
                    side_effect=lambda *args, **kwargs: events.append("examples") or {},
                ) as save_examples,
                mock.patch.object(main, "save_history_plot"),
                mock.patch.object(main, "seed_everything"),
                mock.patch.object(main, "print_hf_auth_status"),
                mock.patch.object(main.torch.cuda, "is_available", return_value=False),
                mock.patch.object(main, "cleanup_distributed", side_effect=lambda: events.append("cleanup")),
                mock.patch.object(
                    main,
                    "save_deferred_examples_after_distributed_cleanup",
                    side_effect=lambda *args, **kwargs: events.append("deferred") or {},
                ) as deferred_examples,
            ):
                FakeTrainer.step_records = copy.deepcopy(records)
                main.main(
                    output_dir=output_dir,
                    max_epochs=1,
                    example_mode="best",
                    run_test=False,
                    num_examples=2,
                )

        save_examples.assert_called_once()
        deferred_examples.assert_not_called()
        self.assertEqual(events, ["examples", "cleanup"])

    def test_example_sample_indices_are_deterministic_and_non_prefix(self) -> None:
        first = main.select_example_sample_indices(
            100,
            num_examples=10,
            seed=42,
            split="validation",
            epoch=1,
        )
        second = main.select_example_sample_indices(
            100,
            num_examples=10,
            seed=42,
            split="validation",
            epoch=1,
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first or ()), 10)
        self.assertEqual(tuple(sorted(first or ())), first)
        self.assertNotEqual(first, tuple(range(10)))

    def test_full_split_examples_use_prepared_population_for_sampling(self) -> None:
        trainer = mock.Mock()
        trainer.seed = 42
        sample_indices = (3, 11, 18, 24, 32, 45, 51, 67, 80, 99)
        example_loader = [mock.Mock()]

        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            mock.patch.object(main, "build_loader") as build_loader,
            mock.patch.object(
                main,
                "build_example_loader",
                return_value=(example_loader, 100, sample_indices),
            ) as build_example_loader,
            mock.patch.object(main, "_render_restoration_examples", return_value=[]) as render,
        ):
            main.save_restoration_examples(
                trainer,
                split="validation",
                max_samples=None,
                epoch=1,
                output_dir=Path(tmp_dir),
                num_examples=10,
            )

        build_loader.assert_not_called()
        build_example_loader.assert_called_once()
        self.assertIsNone(build_example_loader.call_args.kwargs["max_samples"])
        self.assertEqual(build_example_loader.call_args.kwargs["num_examples"], 10)
        self.assertEqual(render.call_args.kwargs["dataloader"], example_loader)
        self.assertIsNone(render.call_args.kwargs["sample_indices"])

    def test_example_loader_reads_only_selected_blocks(self) -> None:
        trainer = mock.Mock()
        trainer.seed = 42
        trainer.streaming = True
        trainer.batch_size = 4
        trainer.include_metadata = True
        blocks = [
            {"cache_key": "b0", "path": "b0.crpack", "row_count": 4},
            {"cache_key": "b1", "path": "b1.crpack", "row_count": 4},
            {"cache_key": "b2", "path": "b2.crpack", "row_count": 4},
        ]
        catalog = {"blocks": blocks}
        sample_plan = mock.Mock()
        sample_plan.selected_blocks.tolist.return_value = [0, 1, 2]
        sample_plan.effective_rows = 12
        rows_by_key = {
            "b0": [{"id": index} for index in range(4)],
            "b1": [{"id": index + 4} for index in range(4)],
            "b2": [{"id": index + 8} for index in range(4)],
        }
        reader = mock.Mock()
        reader.load_block.side_effect = lambda cache_key: rows_by_key[cache_key]
        collate = mock.Mock(side_effect=lambda rows: {"ids": [row["id"] for row in rows]})

        with (
            mock.patch.object(main, "load_hf_v2_manifest"),
            mock.patch.object(main, "load_hf_v2_split_catalog", return_value=catalog),
            mock.patch.object(main, "plan_sample", return_value=sample_plan),
            mock.patch.object(main, "HFV2StagedBlockReader", return_value=reader),
            mock.patch.object(main, "build_collate_fn", return_value=collate),
        ):
            batches, population_size, sample_indices = main.build_example_loader(
                trainer,
                split="validation",
                max_samples=None,
                epoch=1,
                num_examples=2,
                sample_indices=(1, 9),
            )

        prepared_blocks = reader.prepare_blocks.call_args.args[0]
        self.assertEqual([block["cache_key"] for block in prepared_blocks], ["b0", "b2"])
        self.assertEqual([call.args[0] for call in reader.load_block.call_args_list], ["b0", "b2"])
        self.assertEqual(population_size, 12)
        self.assertEqual(sample_indices, (1, 9))
        self.assertEqual(batches, [{"ids": [1, 9]}])

    def test_render_examples_uses_selected_sample_indices(self) -> None:
        trainer = mock.Mock()
        trainer.model = nn.Linear(1, 1)
        trainer.predict.side_effect = lambda batch: batch["cloudy"]
        batch = {
            "sar": torch.zeros(6, 2, 2, 2),
            "cloudy": torch.zeros(6, 13, 2, 2),
            "target": torch.ones(6, 13, 2, 2),
            "meta": {
                "season": ["s0", "s1", "s2", "s3", "s4", "s5"],
                "scene": ["0", "1", "2", "3", "4", "5"],
                "patch": ["p0", "p1", "p2", "p3", "p4", "p5"],
            },
        }

        def save_figure(*, output_dir, split_label, example_index, title, panels):
            del split_label, title, panels
            return output_dir / f"example_{example_index}.png"

        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            mock.patch.object(main, "build_example_panels", return_value=(("panel", torch.zeros(2, 2), None),)),
            mock.patch.object(main, "_save_example_figure", side_effect=save_figure) as save_mock,
        ):
            paths = main._render_restoration_examples(
                trainer,
                [batch],
                output_dir=Path(tmp_dir),
                num_examples=2,
                split_label="validation",
                sample_indices=(1, 4),
            )

        self.assertEqual(len(paths), 2)
        titles = [call.kwargs["title"] for call in save_mock.call_args_list]
        self.assertIn("scene_1/patch_p1", titles[0])
        self.assertIn("scene_4/patch_p4", titles[1])

    def test_resume_uses_trainer_load_checkpoint(self) -> None:
        records = [
            make_epoch_record(epoch=1, train_loss=1.0, val_loss=0.6),
            make_epoch_record(epoch=2, train_loss=0.9, val_loss=0.5),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            resume_path = output_dir / "resume.pt"
            resume_path.write_text("checkpoint:1\n", encoding="utf-8")
            (output_dir / "metrics.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"kind": "train_epoch", "epoch": 1, "loss": 1.0, "metrics": {}}),
                        json.dumps({"kind": "validation", "epoch": 1, "loss": 0.6, "metrics": {"mae": 0.3}}),
                        json.dumps({"kind": "train_epoch", "epoch": 3, "loss": 0.7, "metrics": {}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

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
            history = result["history_plot"].call_args.args[0]
            self.assertEqual([row["epoch"] for row in history], [1, 2])
            self.assertEqual(history[0]["val_mae"], 0.3)
            self.assertEqual(history[1]["val_loss"], 0.5)

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
        trainer.streaming = True
        trainer.dataset_root = Path("/tmp/dataset-root")
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

    def test_build_loader_prepares_streaming_split(self) -> None:
        trainer = self._make_trainer()
        prepared = mock.Mock(num_examples=7176)
        loader = mock.Mock()

        with (
            mock.patch.object(main, "prepare_split", return_value=prepared) as prepare_split,
            mock.patch.object(main, "build_dataloader", return_value=loader) as build_dataloader,
        ):
            result = main.build_loader(
                trainer,
                split="validation",
                max_samples=None,
                training=False,
                epoch_index=3,
            )

        self.assertEqual(result, loader)
        self.assertEqual(loader._cr_prepared_num_examples, 7176)
        self.assertTrue(prepare_split.call_args.kwargs["streaming"])
        self.assertIsNone(prepare_split.call_args.kwargs["dataset_root"])
        self.assertEqual(prepare_split.call_args.kwargs["max_samples"], None)
        self.assertEqual(build_dataloader.call_args.kwargs["crop_size"], None)
        self.assertEqual(build_dataloader.call_args.kwargs["crop_mode"], "none")

    def test_build_loader_passes_training_options(self) -> None:
        trainer = self._make_trainer()
        prepared = mock.Mock(num_examples=107143)
        loader = mock.Mock()

        with (
            mock.patch.object(main, "prepare_split", return_value=prepared) as prepare_split,
            mock.patch.object(main, "build_dataloader", return_value=loader),
        ):
            main.build_loader(
                trainer,
                split="train",
                max_samples=256,
                training=True,
                epoch_index=1,
            )

        self.assertTrue(prepare_split.call_args.kwargs["streaming"])
        self.assertEqual(prepare_split.call_args.kwargs["max_samples"], 256)
        self.assertEqual(loader._cr_prepared_num_examples, 107143)

    def test_build_loader_uses_local_dataset_dir_when_not_streaming(self) -> None:
        trainer = self._make_trainer()
        trainer.streaming = False
        trainer.dataset_root = Path("/tmp/dataset-root")
        prepared = mock.Mock(num_examples=3588)
        loader = mock.Mock()

        with (
            mock.patch.object(main, "prepare_split", return_value=prepared) as prepare_split,
            mock.patch.object(main, "build_dataloader", return_value=loader),
        ):
            result = main.build_loader(
                trainer,
                split="test",
                max_samples=128,
                training=False,
                epoch_index=2,
            )

        self.assertEqual(result, loader)
        self.assertEqual(loader._cr_prepared_num_examples, 3588)
        self.assertFalse(prepare_split.call_args.kwargs["streaming"])
        self.assertEqual(prepare_split.call_args.kwargs["dataset_root"], Path("/tmp/dataset-root"))


class TrainScriptTest(unittest.TestCase):
    def test_multi_gpu_defaults_nccl_p2p_disable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            target = tmp_path / "target.py"
            target.write_text("print('target')\n", encoding="utf-8")
            capture = tmp_path / "capture.txt"
            fake_uv = tmp_path / "uv"
            fake_uv.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "echo \"NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE-}\" > \"${TRAIN_SH_CAPTURE}\"",
                        "printf 'args=%s\\n' \"$*\" >> \"${TRAIN_SH_CAPTURE}\"",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            fake_uv.chmod(0o755)
            env = {
                **os.environ,
                "UV": str(fake_uv),
                "TRAIN_TARGET": str(target),
                "TRAIN_SH_CAPTURE": str(capture),
            }
            env.pop("NCCL_P2P_DISABLE", None)

            subprocess.run(
                ["bash", "train.sh", "--gpus", "5,6"],
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                check=True,
            )

            output = capture.read_text(encoding="utf-8")
        self.assertIn("NCCL_P2P_DISABLE=1", output)
        self.assertIn("--nproc-per-node=2", output)

    def test_multi_gpu_respects_existing_nccl_p2p_disable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            target = tmp_path / "target.py"
            target.write_text("print('target')\n", encoding="utf-8")
            capture = tmp_path / "capture.txt"
            fake_uv = tmp_path / "uv"
            fake_uv.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "echo \"NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE-}\" > \"${TRAIN_SH_CAPTURE}\"",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            fake_uv.chmod(0o755)
            env = {
                **os.environ,
                "UV": str(fake_uv),
                "TRAIN_TARGET": str(target),
                "TRAIN_SH_CAPTURE": str(capture),
                "NCCL_P2P_DISABLE": "0",
            }

            subprocess.run(
                ["bash", "train.sh", "--gpus", "5,6"],
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                check=True,
            )

            output = capture.read_text(encoding="utf-8")
        self.assertIn("NCCL_P2P_DISABLE=0", output)


if __name__ == "__main__":
    unittest.main()
