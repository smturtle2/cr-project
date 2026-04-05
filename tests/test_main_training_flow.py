from __future__ import annotations

import copy
import json
import tempfile
import unittest
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
) -> dict[str, object]:
    return {
        "epoch": epoch,
        "train": {"loss": train_loss, "metrics": {}},
        "val": {"loss": val_loss, "metrics": dict(val_metrics or {})},
        "elapsed_sec": 1.0,
    }


class FakeTrainer:
    step_records: list[dict[str, object]] = []
    test_record: dict[str, object] = {"loss": 0.25, "metrics": {"mae": 0.1}}
    instances: list["FakeTrainer"] = []

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss,
        metrics,
        max_train_samples,
        max_val_samples,
        max_test_samples,
        output_dir,
        batch_size,
        epochs,
        seed,
    ) -> None:
        del loss, metrics, max_train_samples, max_val_samples, max_test_samples
        self.model = model
        self.optimizer = optimizer
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed
        self.num_workers = 0
        self.cache_root = self.output_dir / "cache"
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
        checkpoint_path = self.output_dir / f"epoch-{epoch:04d}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(f"checkpoint:{epoch}\n", encoding="utf-8")
        record["checkpoint_path"] = str(checkpoint_path)
        self.current_epoch = epoch
        self.global_step = epoch * 10
        return record

    def test(self) -> dict[str, object]:
        self.test_calls += 1
        return copy.deepcopy(type(self).test_record)


class MainTrainingFlowTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeTrainer.instances = []

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
            self.assertEqual(result["build_loader"].call_count, 2)
            first_call = result["build_loader"].call_args_list[0]
            second_call = result["build_loader"].call_args_list[1]
            self.assertEqual(first_call.kwargs["split"], "validation")
            self.assertEqual(first_call.kwargs["epoch_index"], 0)
            self.assertEqual(second_call.kwargs["split"], "validation")
            self.assertEqual(second_call.kwargs["epoch_index"], 1)

            self.assertEqual(result["save_examples"].call_count, 2)
            first_output_dir = result["save_examples"].call_args_list[0].kwargs["output_dir"]
            second_output_dir = result["save_examples"].call_args_list[1].kwargs["output_dir"]
            self.assertEqual(first_output_dir, output_dir / "examples" / "best" / "epoch_001")
            self.assertEqual(second_output_dir, output_dir / "examples" / "best" / "epoch_002")
            self.assertEqual(result["save_examples"].call_args_list[0].kwargs["stage"], "val")
            self.assertEqual(result["save_examples"].call_args_list[1].kwargs["stage"], "val")

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
            self.assertEqual(result["build_loader"].call_count, 1)
            self.assertEqual(result["build_loader"].call_args.kwargs["split"], "test")
            self.assertEqual(result["build_loader"].call_args.kwargs["epoch_index"], 1)
            self.assertEqual(result["save_examples"].call_count, 1)
            self.assertEqual(result["save_examples"].call_args.kwargs["stage"], "test")
            self.assertEqual(
                result["save_examples"].call_args.kwargs["output_dir"],
                output_dir / "examples" / "test",
            )

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

    def _run_main(
        self,
        *,
        output_dir: Path,
        step_records: list[dict[str, object]],
        example_mode: str,
        num_examples: int,
        run_test: bool | None = False,
        selector: main.BestEpochSelector | None = None,
    ) -> dict[str, object]:
        FakeTrainer.step_records = copy.deepcopy(step_records)
        FakeTrainer.instances = []
        FakeTrainer.test_record = {"loss": 0.25, "metrics": {"mae": 0.1}}

        build_loader = mock.Mock(side_effect=lambda *args, **kwargs: {"loader": kwargs})
        save_examples = mock.Mock(return_value=[])
        history_plot = mock.Mock()

        model = nn.Linear(1, 1)
        if selector is None:
            selector = main.build_best_epoch_selector()

        with (
            mock.patch.object(main, "Trainer", FakeTrainer),
            mock.patch.object(main, "build_model", return_value=model),
            mock.patch.object(main, "build_optimizer", side_effect=lambda current_model: torch.optim.SGD(current_model.parameters(), lr=0.1)),
            mock.patch.object(main, "build_loss", return_value=lambda prediction, batch: torch.tensor(0.0)),
            mock.patch.object(main, "build_metrics", return_value={}),
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
            }
            if run_test is not None:
                kwargs["run_test"] = run_test

            main.main(
                **kwargs,
            )

        return {
            "trainer": FakeTrainer.instances[-1],
            "build_loader": build_loader,
            "save_examples": save_examples,
            "history_plot": history_plot,
        }


if __name__ == "__main__":
    unittest.main()
